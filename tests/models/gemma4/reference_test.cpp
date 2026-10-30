// The scalar reference on a tiny 26B-A4B-shaped model (tiny_model.hpp)
// against fixtures/reference_tiny.json, the logits of an independent float64
// port of llama.cpp's gemma4 graph (tools/gemma4/reference_oracle.py). The
// tokens run past the sliding window, so window masking, K=V on the global
// layer, rope frequency divisors, routing and the softcap all matter.
//
// TODO(G4-12, nimo): compare the reference with the device runtime on the
// real 26B-A4B artifact; this host test only pins the operator semantics.

#include "src/models/gemma4/reference.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/tiny_model.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

// Largest |reference - oracle| over all logits. Both accumulate in double;
// the reference rounds intermediate activations to float.
constexpr double kTolerance = 1e-4;

gufo::json::Value ReadFixture() {
  std::ifstream file(std::string(GUFO_GEMMA4_FIXTURES) +
                     "/reference_tiny.json");
  Require(file.good(), "missing fixture reference_tiny.json");
  std::stringstream text;
  text << file.rdbuf();
  return gufo::json::parse(text.str());
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    gemma4_test::GgufImage image(gemma4_test::tiny::TinyMetadata(),
                                 gemma4_test::tiny::TinyTensors());
    Require(image.reader() != nullptr, "tiny GGUF: " + image.error());
    std::string error;
    auto weights = g4::ModelWeights::Bind(*image.reader(), &error);
    Require(weights.has_value(), "tiny weights rejected: " + error);
    Require(weights->ReadHostValues(&error), "host values: " + error);

    const auto fixture = ReadFixture();
    std::vector<g4::TokenId> tokens;
    for (const auto& t : fixture.find("tokens")->items()) {
      tokens.push_back(static_cast<g4::TokenId>(t.as_size()));
    }
    const std::size_t vocab = weights->vocab_size;
    const auto& rows = fixture.find("logits")->items();
    Require(rows.size() == tokens.size() && tokens.size() > 4,
            "fixture runs past the window");

    g4::Reference reference(*weights);
    std::vector<float> logits;
    reference.Forward(tokens, &logits, nullptr);
    Require(logits.size() == tokens.size() * vocab, "logit rows");
    double worst = 0.0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      const auto& expected = rows[r].items();
      Require(expected.size() == vocab, "fixture vocabulary");
      for (std::size_t v = 0; v < vocab; ++v) {
        worst = std::max(
            worst, std::fabs(logits[r * vocab + v] - expected[v].as_double()));
      }
    }
    std::cout << "max |reference - oracle| = " << worst << "\n";
    Require(worst <= kTolerance,
            "reference differs from the llama.cpp oracle by " +
                std::to_string(worst));

    // A prefix processed in two calls gives identical results.
    reference.Reset();
    std::vector<float> first;
    std::vector<float> second;
    const std::size_t split = 6;
    reference.Forward(std::span(tokens).first(split), &first, nullptr);
    reference.Forward(std::span(tokens).subspan(split), &second, nullptr);
    first.insert(first.end(), second.begin(), second.end());
    Require(first == logits && reference.Position() == tokens.size(),
            "incremental forward matches a single pass");
  });
}
