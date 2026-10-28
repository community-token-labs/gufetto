#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace gufo::models::qwen38_flash_next::rocm {

namespace {

std::atomic<bool> process_failed{false};
constexpr std::uint64_t kOpcode = 3;
constexpr unsigned kDefaultRunTimeoutMs = 8000;  // a dispatch takes 1-15 ms
constexpr std::size_t kBytesPer8 = 9;
constexpr std::int64_t kHeartbeatQuietMs = 2000;
constexpr std::size_t kMaxInFlight = 4;  // the driver's stock queue depth
constexpr unsigned kCopyHelpers = 3;
constexpr std::size_t kParallelCopyMin = std::size_t{1} << 20;

std::int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// GUFO_NPU_TRACE=<path>: one line per NPU-path event, written through O_DSYNC
// so the last line survives a hang or a freeze.
void Trace(const char* what, long a = 0, long b = 0) {
  static const int fd = [] {
    const char* path = std::getenv("GUFO_NPU_TRACE");
    return path != nullptr && path[0] != '\0'
               ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_DSYNC, 0644)
               : -1;
  }();
  if (fd < 0) {
    return;
  }
  char line[160];
  const int n = std::snprintf(line, sizeof line, "%lld %s %ld %ld\n",
                              static_cast<long long>(NowMs()), what, a, b);
  if (n > 0) {
    (void)!write(fd, line, static_cast<std::size_t>(n));
  }
}

struct Call {
  void* impl;
  int a_id;
  std::vector<int> b_ids;
  std::vector<int> c_ids;
  std::uint64_t ticket;
  int program{0};
};

class CopyPool {
public:
  explicit CopyPool(unsigned helpers) : parts_(helpers + 1), next_(parts_) {
    for (unsigned i = 0; i < helpers; ++i) {
      threads_.emplace_back([this] { Run(); });
    }
  }

  ~CopyPool() {
    {
      const std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }

  void Copy(void* dst, const void* src, std::size_t bytes) {
    if (bytes < kParallelCopyMin) {
      std::memcpy(dst, src, bytes);
      return;
    }
    const std::lock_guard serial(call_mutex_);
    {
      const std::lock_guard lock(mutex_);
      dst_ = static_cast<char*>(dst);
      src_ = static_cast<const char*>(src);
      bytes_ = bytes;
      next_ = 0;
      remaining_ = parts_;
    }
    wake_.notify_all();
    Work();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
  }

private:
  void Work() {
    for (;;) {
      unsigned part = 0;
      char* dst = nullptr;
      const char* src = nullptr;
      std::size_t begin = 0;
      std::size_t end = 0;
      {
        const std::lock_guard lock(mutex_);
        if (next_ >= parts_) {
          return;
        }
        part = next_++;
        const std::size_t chunk = (bytes_ + parts_ - 1) / parts_;
        begin = std::min<std::size_t>(part * chunk, bytes_);
        end = std::min(begin + chunk, bytes_);
        dst = dst_;
        src = src_;
      }
      std::memcpy(dst + begin, src + begin, end - begin);
      const std::lock_guard lock(mutex_);
      if (--remaining_ == 0) {
        done_.notify_all();
      }
    }
  }

  void Run() {
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this] { return stop_ || next_ < parts_; });
        if (stop_) {
          return;
        }
      }
      Work();
    }
  }

  const unsigned parts_;
  std::mutex call_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  char* dst_{nullptr};
  const char* src_{nullptr};
  std::size_t bytes_{0};
  unsigned next_;
  unsigned remaining_{0};
  bool stop_{false};
  std::vector<std::thread> threads_;
};

}  // namespace

struct NpuGemm::Impl {
  Config config;
  xrt::device device;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo instructions;
  std::size_t instruction_words{0};
  int group_a{0};
  /// Further instruction streams for the same kernel (other reduction lengths
  /// of a runtime-K program). Index 0 of a call's `program` is the one above.
  struct Program {
    xrt::bo instructions;
    std::size_t words{0};
  };
  std::vector<Program> extra_programs;

  struct Buffer {
    void* host{nullptr};    // GPU-visible pinned memory
    void* device{nullptr};  // the same memory as a GPU pointer
    std::size_t bytes{0};
    xrt::bo shared_bo;   // userptr import of `host` (normal mode)
    xrt::bo private_bo;  // NPU-only pages (isolate mode)
    void* private_host{nullptr};
  };
  std::vector<Buffer> buffers;
  // Private XRT BOs avoid kernel mmap EINVAL on imported HIP pages.
  const bool isolate{[] {
    const char* value = std::getenv("GUFO_NPU_ISOLATE");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }()};
  std::mutex api_mutex;
  std::vector<hipStream_t> streams;
  void TrackStream(hipStream_t stream) {
    if (std::find(streams.begin(), streams.end(), stream) == streams.end()) {
      streams.push_back(stream);
    }
  }

  CopyPool copy_pool{kCopyHelpers};

  std::mutex mutex;
  std::deque<std::pair<xrt::run, int>> pending;  // run, its C buffer

  // Submissions run on a worker so the stream only blocks in Wait.
  std::thread worker;
  std::mutex queue_mutex;
  std::condition_variable queue_wake;
  std::condition_variable queue_idle;
  std::deque<Call> queue;
  std::uint64_t enqueued{0};
  std::atomic<std::uint64_t> submitted{0};
  std::uint64_t completed{0};
  bool worker_stop{false};
  std::atomic<bool> failed{false};
  std::atomic<std::int64_t> last_submit_ms{0};  // steady clock, ms
  std::chrono::milliseconds run_timeout{kDefaultRunTimeoutMs};
  std::vector<xrt::run>
      abandoned;  // timed-out runs stay alive: never destroyed under the NPU
  std::string last_error;

  // GUFO_NPU_STATS=1: wall-clock accounting printed every 36 waits.
  const bool stats{std::getenv("GUFO_NPU_STATS") != nullptr};
  double submit_us{0};
  // Phase totals of the worker (GUFO_NPU_STATS): where the NPU path spends
  // time.
  double copy_a_us{0};    // A host->private copy + sync
  double launch_us{0};    // kernel(...) submissions
  double run_wait_us{0};  // waiting for the NPU itself
  double copy_c_us{0};    // C sync + private->host copy
  double wait_us{0};
  std::uint64_t waits{0};

  // Keepalive: the driver suspends the NPU after 5 s without a job, and waking
  // it re-initializes the SMU, PSP and mailbox while the GPU may be running
  // flat out. A tiny job every second keeps it powered for the object's life.
  std::thread keepalive;
  std::mutex keepalive_mutex;
  std::condition_variable keepalive_wake;
  bool keepalive_stop{false};
  int heartbeat_a{-1};
  int heartbeat_b{-1};
  int heartbeat_c{-1};

  ~Impl() {
    StopKeepalive();
    StopWorker();
    // Failed instances are retained whole by NpuGemm's destructor.
    pending.clear();
    for (auto& b : buffers) {
      b.shared_bo = xrt::bo();
      b.private_bo = xrt::bo();
      if (b.host != nullptr) {
        (void)hipHostFree(b.host);
      }
    }
  }

  xrt::bo& NpuBo(int id) {
    Buffer& b = buffers[id];
    return isolate ? b.private_bo : b.shared_bo;
  }

  static double Since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t)
        .count();
  }

  void CopyToNpu(int id) {
    const auto t0 = std::chrono::steady_clock::now();
    Buffer& b = buffers[id];
    if (isolate) {
      copy_pool.Copy(b.private_host, b.host, b.bytes);
    }
    NpuBo(id).sync(XCL_BO_SYNC_BO_TO_DEVICE);
    copy_a_us += Since(t0);
  }

  void CopyFromNpu(int id) {
    const auto t0 = std::chrono::steady_clock::now();
    Buffer& b = buffers[id];
    NpuBo(id).sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (isolate) {
      copy_pool.Copy(b.host, b.private_host, b.bytes);
    }
    copy_c_us += Since(t0);
  }

  void Fail(const std::exception& e) {
    process_failed = true;
    failed = true;
    last_error = e.what();
    std::fprintf(stderr, "NPU: %s\n", last_error.c_str());
  }

  // Waits for `run` with a bound; a timeout poisons this object instead of
  // blocking the worker (and through it the HIP stream) forever.
  bool WaitRun(xrt::run& run) {
    const ert_cmd_state state = run.wait(run_timeout);
    if (state == ERT_CMD_STATE_COMPLETED) {
      return true;
    }
    process_failed = true;
    failed = true;
    last_error = state == ERT_CMD_STATE_TIMEOUT
                     ? "NPU run timed out after " +
                           std::to_string(run_timeout.count()) + " ms"
                     : "NPU run did not complete";
    std::fprintf(stderr, "NPU: %s\n", last_error.c_str());
    return false;
  }

  // Waits for the oldest outstanding run (caller holds `mutex`).
  void ReapOldest() {
    auto& run = pending.front().first;
    const int c = pending.front().second;
    Trace("reap_begin", c);
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = WaitRun(run);
    run_wait_us += Since(t0);
    if (!ok) {
      abandoned.push_back(std::move(run));
    }
    pending.pop_front();
    Trace("reap_done", c, ok);
    if (ok) {
      CopyFromNpu(c);
    }
  }

  bool Heartbeat() {
    const std::lock_guard lock(mutex);
    if (failed || heartbeat_a < 0) {
      return false;
    }
    if (!pending.empty()) {
      return true;  // real jobs are outstanding: the device is awake
    }
    // A prefill submits every layer; the heartbeat only matters in a real lull.
    if (NowMs() - last_submit_ms.load() < kHeartbeatQuietMs) {
      return true;
    }
    try {
      Trace("heartbeat_fire");
      pending.emplace_back(xrt::run(kernel), heartbeat_c);
      auto& run = pending.back().first;
      run.set_arg(0, kOpcode);
      run.set_arg(1, instructions);
      run.set_arg(2, instruction_words);
      run.set_arg(3, NpuBo(heartbeat_a));
      run.set_arg(4, NpuBo(heartbeat_b));
      run.set_arg(5, NpuBo(heartbeat_c));
      run.start();
      ReapOldest();
      return !failed;
    } catch (const std::exception& e) {
      Fail(e);
      return false;
    }
  }

  void StopKeepalive() {
    {
      const std::lock_guard lock(keepalive_mutex);
      keepalive_stop = true;
    }
    keepalive_wake.notify_all();
    if (keepalive.joinable()) {
      keepalive.join();
    }
  }

  void StopWorker() {
    {
      const std::lock_guard lock(queue_mutex);
      worker_stop = true;
    }
    queue_wake.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void WorkerLoop() {
    std::unique_lock lock(queue_mutex);
    for (;;) {
      queue_wake.wait(lock, [this] { return worker_stop || !queue.empty(); });
      if (queue.empty()) {
        return;
      }
      Call call = std::move(queue.front());
      queue.pop_front();
      lock.unlock();
      Process(call);
      lock.lock();
      ++completed;
      queue_idle.notify_all();
    }
  }

  void Enqueue(Call call) {
    Trace("enqueue", call.a_id);
    {
      const std::lock_guard lock(queue_mutex);
      queue.push_back(std::move(call));
      ++enqueued;
    }
    queue_wake.notify_one();
  }

  void Process(const Call& call) {
    Trace("process_wait_lock", call.a_id);
    const std::lock_guard lock(mutex);
    Trace("process_begin", call.a_id, static_cast<long>(call.b_ids.size()));
    const auto t0 = std::chrono::steady_clock::now();
    last_submit_ms = NowMs();
    if (failed) {
      return;
    }
    try {
      CopyToNpu(call.a_id);
      for (std::size_t i = 0; i < call.b_ids.size() && !failed; ++i) {
        while (pending.size() >= kMaxInFlight && !failed) {
          ReapOldest();
        }
        if (failed) {
          break;
        }
        const auto t_launch = std::chrono::steady_clock::now();
        const auto& program_bo =
            call.program == 0 ? instructions
                              : extra_programs[call.program - 1].instructions;
        const std::size_t program_words =
            call.program == 0 ? instruction_words
                              : extra_programs[call.program - 1].words;
        // Retain before start: an exception need not mean DMA has stopped.
        pending.emplace_back(xrt::run(kernel), call.c_ids[i]);
        auto& run = pending.back().first;
        run.set_arg(0, kOpcode);
        run.set_arg(1, program_bo);
        run.set_arg(2, program_words);
        run.set_arg(3, NpuBo(call.a_id));
        run.set_arg(4, NpuBo(call.b_ids[i]));
        run.set_arg(5, NpuBo(call.c_ids[i]));
        run.start();
        launch_us += Since(t_launch);
      }
      while (!pending.empty() && !failed) {
        ReapOldest();
      }
    } catch (const std::exception& e) {
      Fail(e);
      // Preserve pending runs and their resources on uncertain completion.
    }
    submit_us += std::chrono::duration<double, std::micro>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    Trace("process_end", call.a_id);
  }

  void WaitNow(std::uint64_t ticket = 0) {
    Trace("wait_begin");
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::unique_lock lock(queue_mutex);
      queue_idle.wait(lock, [this, ticket] {
        return ticket != 0 ? completed >= ticket : completed == enqueued;
      });
    }
    wait_us += std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    Trace("wait_end");
    if (stats && ++waits % 36 == 0) {
      std::fprintf(stderr,
                   "npu stats: %llu layers | worker %.0f us (A copy %.0f, "
                   "launch %.0f, NPU wait %.0f, C copy %.0f), stream blocked "
                   "in wait %.0f us (per layer averages)\n",
                   static_cast<unsigned long long>(waits), submit_us / waits,
                   copy_a_us / waits, launch_us / waits, run_wait_us / waits,
                   copy_c_us / waits, wait_us / waits);
    }
  }
};

NpuGemm::NpuGemm() = default;

bool NpuGemm::ReleaseStream(hipStream_t stream) {
  const std::lock_guard lock(impl_->api_mutex);
  if (hipStreamSynchronize(stream) != hipSuccess) {
    impl_->failed = true;
    return false;
  }
  std::erase(impl_->streams, stream);
  return Healthy();
}

NpuGemm::~NpuGemm() {
  if (!impl_) {
    return;
  }
  // The caller must not destroy this object concurrently with a public call.
  for (hipStream_t stream : impl_->streams) {
    if (hipStreamSynchronize(stream) != hipSuccess) {
      impl_->failed = true;
    }
  }
  impl_->StopKeepalive();
  impl_->StopWorker();
  if (impl_->failed) {
    process_failed = true;
    // Quarantine the entire context, instructions, runs and data until process
    // exit. Abort/destruction is not bounded and is not proof DMA has stopped.
    (void)impl_.release();
  }
}

std::unique_ptr<NpuGemm> NpuGemm::Create(const Config& config,
                                        std::string* error) {
  auto fail = [&](const std::string& message) -> std::unique_ptr<NpuGemm> {
    if (error != nullptr) {
      *error = message;
    }
    return nullptr;
  };
  if (process_failed) {
    return fail("NPU process is quarantined; restart before further dispatch");
  }
  if (config.tokens == 0 || config.k == 0 || config.n == 0 ||
      config.tokens % 8 != 0 || config.k % 8 != 0 || config.n % 8 != 0) {
    return fail("NPU dimensions must be nonzero multiples of eight");
  }
  const auto fits = [](std::uint32_t a, std::uint32_t b) {
    return std::uint64_t{a} * b / 8 <=
           std::numeric_limits<std::size_t>::max() / kBytesPer8;
  };
  if (!fits(config.tokens, config.k) || !fits(config.k, config.n) ||
      !fits(config.tokens, config.n)) {
    return fail("NPU operand size overflow");
  }
  std::unique_ptr<NpuGemm> gemm(new NpuGemm());
  gemm->impl_ = std::make_unique<Impl>();
  Impl& i = *gemm->impl_;
  i.config = config;
  try {
    i.device = xrt::device(0u);
    const xrt::xclbin xclbin{std::string(config.xclbin)};
    i.device.register_xclbin(xclbin);
    i.context = xrt::hw_context(i.device, xclbin.get_uuid());
    i.kernel = xrt::kernel(i.context, xclbin.get_kernels().at(0).get_name());
    std::ifstream file(config.insts, std::ios::binary | std::ios::ate);
    if (!file) {
      return fail("NPU instruction file not readable: " + config.insts);
    }
    const std::streamoff end = file.tellg();
    if (end <= 0 || end > 64 * 1024 * 1024 || end % 4 != 0) {
      return fail("NPU instruction file must contain 1..64 MiB of whole words");
    }
    const auto size = static_cast<std::size_t>(end);
    file.seekg(0);
    std::vector<char> words(size);
    if (!file.read(words.data(), static_cast<std::streamsize>(size))) {
      return fail("NPU instruction file read failed: " + config.insts);
    }
    i.instruction_words = size / sizeof(int);
    i.instructions =
        xrt::bo(i.device, size, XCL_BO_FLAGS_CACHEABLE, i.kernel.group_id(1));
    std::memcpy(i.instructions.map<void*>(), words.data(), size);
    i.instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    i.group_a = i.kernel.group_id(3);
  } catch (const std::exception& e) {
    return fail(std::string("NPU setup failed: ") + e.what());
  }
  if (const char* t = std::getenv("GUFO_NPU_TIMEOUT_MS"); t != nullptr) {
    i.run_timeout = std::chrono::milliseconds(std::max(1, std::atoi(t)));
  }
  // Wake the NPU now, while the GPU is still light, and keep it awake.
  i.heartbeat_a = gemm->NewBuffer(gemm->ABytes(), error);
  i.heartbeat_b = gemm->NewBuffer(gemm->BBytes(), error);
  i.heartbeat_c = gemm->NewBuffer(gemm->CBytes(), error);
  if (i.heartbeat_a < 0 || i.heartbeat_b < 0 || i.heartbeat_c < 0) {
    return nullptr;
  }
  try {
    for (int id : {i.heartbeat_a, i.heartbeat_b}) {
      std::memset(i.buffers[id].host, 0, i.buffers[id].bytes);
      i.CopyToNpu(id);
    }
  } catch (const std::exception& e) {
    return fail(std::string("NPU initial buffer sync failed: ") + e.what());
  }
  if (!i.Heartbeat()) {
    return fail("NPU did not run its first job: " + i.last_error);
  }
  Impl* impl = &i;
  i.worker = std::thread([impl] { impl->WorkerLoop(); });
  const char* keepalive_env = std::getenv("GUFO_NPU_KEEPALIVE");
  if (keepalive_env != nullptr && keepalive_env[0] == '0') {
    return gemm;
  }
  i.keepalive = std::thread([impl] {
    std::unique_lock lock(impl->keepalive_mutex);
    while (!impl->keepalive_stop) {
      impl->keepalive_wake.wait_for(lock, std::chrono::milliseconds(1000));
      if (impl->keepalive_stop) {
        break;
      }
      lock.unlock();
      impl->Heartbeat();
      lock.lock();
    }
  });
  return gemm;
}

std::size_t NpuGemm::ABytes() const noexcept {
  return std::size_t{impl_->config.tokens} * impl_->config.k / 8 * kBytesPer8;
}
std::size_t NpuGemm::BBytes() const noexcept {
  return std::size_t{impl_->config.k} * impl_->config.n / 8 * kBytesPer8;
}
std::size_t NpuGemm::CBytes() const noexcept {
  return std::size_t{impl_->config.tokens} * impl_->config.n / 8 * kBytesPer8;
}
std::uint32_t NpuGemm::tokens() const noexcept {
  return impl_->config.tokens;
}
std::uint32_t NpuGemm::k() const noexcept {
  return impl_->config.k;
}
std::uint32_t NpuGemm::n() const noexcept {
  return impl_->config.n;
}

int NpuGemm::NewBuffer(std::size_t bytes, std::string* error) {
  Impl& i = *impl_;
  const std::lock_guard lock(i.mutex);
  if (bytes == 0 || i.failed) {
    if (error != nullptr) {
      *error = "NPU buffer size is zero or runtime is unhealthy";
    }
    return -1;
  }
  void* host = nullptr;
  // Isolate mode wants ordinary cached pages on the CPU side; the GPU path is
  // flushed by kernel boundaries.
  const unsigned flags =
      hipHostMallocMapped | (i.isolate ? hipHostMallocNonCoherent : 0U);
  if (hipHostMalloc(&host, bytes, flags) != hipSuccess) {
    if (error != nullptr) {
      *error = "hipHostMalloc failed for an NPU buffer";
    }
    return -1;
  }
  try {
    Impl::Buffer buffer;
    buffer.host = host;
    buffer.bytes = bytes;
    if (hipHostGetDevicePointer(&buffer.device, host, 0) != hipSuccess) {
      throw std::runtime_error("hipHostGetDevicePointer failed");
    }
    if (i.isolate) {
      buffer.private_bo =
          xrt::bo(i.device, bytes, XRT_BO_FLAGS_HOST_ONLY, i.group_a);
      buffer.private_host = buffer.private_bo.map<void*>();
    } else {
      buffer.shared_bo =
          xrt::bo(i.device, host, bytes, XRT_BO_FLAGS_HOST_ONLY, i.group_a);
    }
    i.buffers.push_back(std::move(buffer));
  } catch (const std::exception& e) {
    (void)hipHostFree(host);
    if (error != nullptr) {
      *error = std::string("NPU buffer setup failed: ") + e.what();
    }
    return -1;
  }
  return static_cast<int>(i.buffers.size()) - 1;
}

bool NpuGemm::Healthy() const noexcept {
  return !process_failed && !impl_->failed;
}

void* NpuGemm::Host(int id) const {
  return impl_->buffers.at(id).host;
}

void* NpuGemm::Device(int id) const {
  return impl_->buffers.at(id).device;
}

void NpuGemm::FinalizeWeights(int id) {
  const std::lock_guard lock(impl_->mutex);
  try {
    (void)impl_->buffers.at(id);
    if (!impl_->failed) {
      impl_->CopyToNpu(id);
    }
  } catch (const std::exception& e) {
    impl_->Fail(e);
  }
}

int NpuGemm::AddProgram(const std::string& insts, std::string* error) {
  const std::lock_guard lock(impl_->mutex);
  try {
    std::ifstream file(insts, std::ios::binary | std::ios::ate);
    if (!file) {
      if (error != nullptr) {
        *error = "NPU instruction file not readable: " + insts;
      }
      return -1;
    }
    const std::streamoff end = file.tellg();
    if (end <= 0 || end > 64 * 1024 * 1024 || end % 4 != 0) {
      throw std::runtime_error("NPU instruction file has invalid word count");
    }
    const auto size = static_cast<std::size_t>(end);
    file.seekg(0);
    std::vector<char> words(size);
    if (!file.read(words.data(), static_cast<std::streamsize>(size))) {
      throw std::runtime_error("NPU instruction file read failed");
    }
    Impl::Program program;
    program.words = size / sizeof(int);
    program.instructions = xrt::bo(impl_->device, size, XCL_BO_FLAGS_CACHEABLE,
                                   impl_->kernel.group_id(1));
    std::memcpy(program.instructions.map<void*>(), words.data(), size);
    program.instructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->extra_programs.push_back(std::move(program));
    return static_cast<int>(impl_->extra_programs.size());
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string("NPU program setup failed: ") + e.what();
    }
    return -1;
  }
}

std::size_t NpuGemm::ABytes(std::uint32_t k) const noexcept {
  return std::size_t{impl_->config.tokens} * k / 8 * kBytesPer8;
}
std::size_t NpuGemm::BBytes(std::uint32_t k) const noexcept {
  return std::size_t{k} * impl_->config.n / 8 * kBytesPer8;
}

bool NpuGemm::Submit(hipStream_t stream, int a_id, const int* b_ids,
                     const int* c_ids, int count, int program) {
  const std::lock_guard api_lock(impl_->api_mutex);
  if (!Healthy() || count <= 0 || b_ids == nullptr || c_ids == nullptr) {
    return false;
  }
  {
    const std::lock_guard lock(impl_->mutex);
    const auto valid = [&](int id, std::size_t bytes) {
      return id >= 0 && static_cast<std::size_t>(id) < impl_->buffers.size() &&
             impl_->buffers[id].bytes >= bytes;
    };
    if (program < 0 ||
        static_cast<std::size_t>(program) > impl_->extra_programs.size() ||
        !valid(a_id, program == 0 ? ABytes() : 1)) {
      return false;
    }
    for (int j = 0; j < count; ++j) {
      if (!valid(b_ids[j], program == 0 ? BBytes() : 1) ||
          !valid(c_ids[j], CBytes()) || c_ids[j] == a_id) {
        return false;
      }
      for (int l = 0; l < count; ++l) {
        if (c_ids[j] == b_ids[l] || (j != l && c_ids[j] == c_ids[l])) {
          return false;
        }
      }
    }
  }
  impl_->TrackStream(stream);
  Trace("submit_host", a_id, count);
  auto* call = new Call{impl_.get(),
                        a_id,
                        {b_ids, b_ids + count},
                        {c_ids, c_ids + count},
                        ++impl_->submitted,
                        program};
  const hipError_t status = hipLaunchHostFunc(
      stream,
      [](void* data) {
        std::unique_ptr<Call> c(static_cast<Call*>(data));
        static_cast<Impl*>(c->impl)->Enqueue(std::move(*c));
      },
      call);
  if (status != hipSuccess) {
    delete call;
    --impl_->submitted;
    impl_->failed = true;
    return false;
  }
  return true;
}

void* NpuGemm::Staged(hipStream_t /*stream*/, int id) {
  return impl_->buffers.at(id).device;
}

std::uint64_t NpuGemm::Ticket() const noexcept {
  return impl_->submitted;
}

bool NpuGemm::Wait(hipStream_t stream, std::uint64_t ticket) {
  const std::lock_guard lock(impl_->api_mutex);
  if (ticket > Ticket()) {
    return false;
  }
  if (ticket == 0) {
    return Healthy();
  }
  impl_->TrackStream(stream);
  struct Req {
    Impl* impl;
    std::uint64_t ticket;
  };
  auto* req = new Req{impl_.get(), ticket};
  if (hipLaunchHostFunc(
          stream,
          [](void* data) {
            std::unique_ptr<Req> r(static_cast<Req*>(data));
            r->impl->WaitNow(r->ticket);
          },
          req) != hipSuccess) {
    delete req;
    impl_->failed = true;
    return false;
  }
  return true;
}

bool NpuGemm::Wait(hipStream_t stream) {
  Trace("wait_host");
  return Wait(stream, Ticket());
}

}  // namespace gufo::models::qwen38_flash_next::rocm
