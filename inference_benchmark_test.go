package onnx

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
)

// BenchmarkInference separates wire parsing, loading and execution on the fixed
// MLP, CNN and shape fixtures. Fixture decoding and comparisons are not timed.
func BenchmarkInference(parent *testing.B) {
	for _, fixture := range inferenceFixtures(parent) {
		parent.Run(fixture.Name, func(group *testing.B) {
			modelData, readErr := fixture.Model.read()
			if readErr != nil {
				group.Fatal(readErr)
			}

			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				group.Fatal(inputErr)
			}

			ctx := context.Background()
			limits, limitErr := (Limits{}).normalized()
			if limitErr != nil {
				group.Fatal(limitErr)
			}

			group.Run("Parse", func(parse *testing.B) {
				parse.ReportAllocs()
				for parse.Loop() {
					if _, parseErr := decode.ParseModel(ctx, modelData, limits.wire()); parseErr != nil {
						parse.Fatal(parseErr)
					}
				}
			})
			group.Run("Load", func(load *testing.B) {
				load.ReportAllocs()
				for load.Loop() {
					model, loadErr := LoadBytes(ctx, modelData, Options{})
					if loadErr != nil {
						load.Fatal(loadErr)
					}

					if loadErr = model.Close(); loadErr != nil {
						load.Fatal(loadErr)
					}
				}
			})
			model, loadErr := LoadBytes(ctx, modelData, Options{})
			if loadErr != nil {
				group.Fatal(loadErr)
			}

			group.Cleanup(func() {
				if closeErr := model.Close(); closeErr != nil {
					group.Error(closeErr)
				}
			})
			group.Run("First", func(first *testing.B) {
				first.ReportAllocs()
				for first.Loop() {
					session, sessionErr := model.NewSession()
					if sessionErr != nil {
						first.Fatal(sessionErr)
					}

					if _, sessionErr = session.Run(ctx, inputs, RunOptions{}); sessionErr != nil {
						first.Fatal(sessionErr)
					}

					if sessionErr = session.Close(); sessionErr != nil {
						first.Fatal(sessionErr)
					}
				}
			})
			group.Run("Warm", func(warm *testing.B) {
				session, sessionErr := model.NewSession()
				if sessionErr != nil {
					warm.Fatal(sessionErr)
				}

				warm.Cleanup(func() {
					if closeErr := session.Close(); closeErr != nil {
						warm.Error(closeErr)
					}
				})
				outputs, runErr := session.Run(ctx, inputs, RunOptions{})
				if runErr != nil {
					warm.Fatal(runErr)
				}

				if runErr = fixture.compareOutputs(outputs); runErr != nil {
					warm.Fatal(runErr)
				}

				warm.ReportAllocs()
				warm.ResetTimer()
				for warm.Loop() {
					if _, runErr = session.Run(ctx, inputs, RunOptions{}); runErr != nil {
						warm.Fatal(runErr)
					}
				}
			})
		})
	}
}

func inferenceFixtures(tb testing.TB) (cases []conformanceCase) {
	tb.Helper()
	for _, path := range []string{"testdata/inference/provenance.json", "testdata/m3/provenance.json"} {
		data, err := os.ReadFile(path)
		if err != nil {
			tb.Fatal(err)
		}

		var suite struct {
			Cases []conformanceCase `json:"cases"`
		}

		if err = json.Unmarshal(data, &suite); err != nil {
			tb.Fatal(err)
		}

		for _, fixture := range suite.Cases {
			if strings.HasPrefix(fixture.Name, "mlp_batch_") || strings.HasPrefix(fixture.Name, "shape_batch_") || strings.HasPrefix(fixture.Name, "cnn_batch_") {
				cases = append(cases, fixture)
			}
		}
	}
	return cases
}
