#ifndef GUFO_CORE_HIP_COMMITTED_BACKING_HPP_
#define GUFO_CORE_HIP_COMMITTED_BACKING_HPP_

#include "src/cache/backing.hpp"

namespace gufo::hip {

using BackingPin = cache::BackingPin;
using BackingBlock = cache::BackingBlock;
using BackingPoolConfig = cache::BackingPoolConfig;

// GPU registration is the only device-specific part of the shared physical
// pool. All assignment, lifetime, growth and trim contracts live in
// backing.hpp.
class CommittedBackingPool : public cache::CommittedBackingPool {
public:
  CommittedBackingPool(cache::ResourceLedger&, BackingPoolConfig);
};

}  // namespace gufo::hip
#endif
