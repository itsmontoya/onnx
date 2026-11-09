// Command compare_runtime runs named tensor cases for the Python comparison driver.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/itsmontoya/onnx"
)

func main() {
	var (
		modelPath  = flag.String("model", "", "ONNX model path")
		warmup     = flag.Int("warmup", 10, "untimed runs per case")
		iterations = flag.Int("iterations", 100, "timed runs per case")
	)

	flag.Parse()
	if err := run(*modelPath, *warmup, *iterations); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(modelPath string, warmup, iterations int) (err error) {
	if modelPath == "" || warmup < 0 || iterations < 1 {
		return fmt.Errorf("model is required; warmup must be nonnegative and iterations positive")
	}

	var (
		cases   map[string][]tensorData
		model   *onnx.Model
		session *onnx.Session
		result  runResult
		names   []string
	)

	if err = json.NewDecoder(os.Stdin).Decode(&cases); err != nil {
		return fmt.Errorf("decode input cases: %w", err)
	}

	if len(cases) == 0 {
		return fmt.Errorf("at least one input case is required")
	}

	ctx := context.Background()
	started := time.Now()
	if model, err = onnx.LoadFile(ctx, modelPath, onnx.Options{}); err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	result.LoadNS = time.Since(started).Nanoseconds()
	defer func() { err = errors.Join(err, model.Close()) }()
	started = time.Now()
	if session, err = model.NewSession(); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	result.SessionNS = time.Since(started).Nanoseconds()
	defer func() { err = errors.Join(err, session.Close()) }()
	result.Outputs = make(map[string][]tensorData, len(cases))
	result.LatenciesNS = make(map[string][]int64, len(cases))
	for name := range cases {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if result.Outputs[name], result.LatenciesNS[name], err = runCase(ctx, session, cases[name], warmup, iterations); err != nil {
			return fmt.Errorf("case %q: %w", name, err)
		}
	}
	if err = json.NewEncoder(os.Stdout).Encode(result); err != nil {
		return fmt.Errorf("encode results: %w", err)
	}

	return nil
}

func runCase(ctx context.Context, session *onnx.Session, values []tensorData, warmup, iterations int) (outputs []tensorData, latencies []int64, err error) {
	var (
		inputs = make(map[string]*onnx.Tensor, len(values))
		result []onnx.NamedTensor
	)

	for _, value := range values {
		if _, exists := inputs[value.Name]; exists {
			return nil, nil, fmt.Errorf("duplicate input %q", value.Name)
		}

		if inputs[value.Name], err = value.tensor(); err != nil {
			return nil, nil, err
		}
	}
	latencies = make([]int64, iterations)
	for index := 0; index < warmup+iterations; index++ {
		started := time.Now()
		if result, err = session.Run(ctx, inputs, onnx.RunOptions{}); err != nil {
			return nil, nil, fmt.Errorf("inference: %w", err)
		}

		if index >= warmup {
			latencies[index-warmup] = time.Since(started).Nanoseconds()
		}
	}
	for _, value := range result {
		var encoded tensorData
		if encoded, err = encodeTensor(value); err != nil {
			return nil, nil, err
		}

		outputs = append(outputs, encoded)
	}
	return outputs, latencies, nil
}
