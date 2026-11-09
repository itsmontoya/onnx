package onnx

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// BenchmarkInference supplies an unchanged copying baseline for NN-13. Timings
// exclude fixture decoding. They do not measure copied bytes or peak live bytes.
func BenchmarkInference(parent *testing.B) {
	data, err := os.ReadFile("testdata/inference/provenance.json")
	if err != nil {
		parent.Fatal(err)
	}

	var suite struct {
		Cases []conformanceCase `json:"cases"`
	}

	if err = json.Unmarshal(data, &suite); err != nil {
		parent.Fatal(err)
	}

	for _, fixture := range suite.Cases {
		if !strings.HasPrefix(fixture.Name, "mlp_") && !strings.HasPrefix(fixture.Name, "shape_") {
			continue
		}

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
