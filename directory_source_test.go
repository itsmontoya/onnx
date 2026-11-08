package onnx

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestDirectorySourceErrors(t *testing.T) {
	directory, resolver := directoryFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	source, err := resolver.Resolve(ctx, "weights.bin", 0, 10)
	if err != nil {
		t.Fatal(err)
	}

	defer source.Close()
	cancel()
	if count, readErr := source.Read(make([]byte, 10)); count != 0 || !errors.Is(readErr, context.Canceled) {
		t.Fatalf("reader cancellation: %d, %v", count, readErr)
	}

	truncated, err := resolver.Resolve(context.Background(), "weights.bin", 0, 10)
	if err != nil {
		t.Fatal(err)
	}

	defer truncated.Close()
	if err = os.Truncate(filepath.Join(directory, "weights.bin"), 3); err != nil {
		t.Fatal(err)
	}

	if count, readErr := io.ReadFull(truncated, make([]byte, 10)); count != 3 || !errors.Is(readErr, io.ErrUnexpectedEOF) {
		t.Fatalf("truncation cause: %d, %v", count, readErr)
	}

	if err = truncated.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err = truncated.Read(make([]byte, 1)); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("closed read: %v", err)
	}

	if err = truncated.Close(); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("close cause: %v", err)
	}
}
