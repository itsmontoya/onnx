package onnx

import (
	"context"
	"errors"
	"os"
	"testing"
)

// BenchmarkMNISTPreparation measures loading independently of inference and
// session creation. The optional model is supplied by tools/compare_runtime.py.
func BenchmarkMNISTPreparation(b *testing.B) {
	var (
		ctx   = context.Background()
		data  []byte
		model *Model
		err   error
	)

	if data, err = os.ReadFile(".cache/compare-runtime/mnist-ir8-opset13.onnx"); errors.Is(err, os.ErrNotExist) {
		b.Skip("prepare the MNIST model with tools/compare_runtime.py")
	} else if err != nil {
		b.Fatal(err)
	}

	b.Run("LoadBytes", func(load *testing.B) {
		load.ReportAllocs()
		for load.Loop() {
			if model, err = LoadBytes(ctx, data, Options{}); err != nil {
				load.Fatal(err)
			}

			if err = model.Close(); err != nil {
				load.Fatal(err)
			}
		}
	})
	if model, err = LoadBytes(ctx, data, Options{}); err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := model.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	b.Run("Session", func(creation *testing.B) {
		var session *Session
		creation.ReportAllocs()
		for creation.Loop() {
			if session, err = model.NewSession(); err != nil {
				creation.Fatal(err)
			}

			if err = session.Close(); err != nil {
				creation.Fatal(err)
			}
		}
	})
}
