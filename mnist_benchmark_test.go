package onnx

import (
	"context"
	"errors"
	"os"
	"testing"
)

// BenchmarkMNIST profiles the optional pretrained model prepared by
// tools/compare_runtime.py. It never downloads fixtures during Go tests.
func BenchmarkMNIST(b *testing.B) {
	const path = ".cache/compare-runtime/mnist-ir8-opset13.onnx"
	var (
		ctx     = context.Background()
		model   *Model
		session *Session
		input   *Tensor
		err     error
	)

	if model, err = LoadFile(ctx, path, Options{}); errors.Is(err, os.ErrNotExist) {
		b.Skip("prepare the MNIST model with tools/compare_runtime.py")
	} else if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := model.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	if session, err = model.NewSession(); err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := session.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})
	values := make([]float32, 28*28)
	for index := range values {
		values[index] = float32(index%256) / 255
	}
	if input, err = NewFloat32([]int64{1, 1, 28, 28}, values); err != nil {
		b.Fatal(err)
	}

	inputs := map[string]*Tensor{"Input3": input}
	if _, err = session.Run(ctx, inputs, RunOptions{}); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err = session.Run(ctx, inputs, RunOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}
