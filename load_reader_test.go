package onnx

import (
	"context"
	"io"
	"testing"
)

type loadReader struct {
	data         []byte
	failure      error
	chunk        int
	reads        int
	consumed     int
	largest      int
	closed       bool
	cancel       context.CancelFunc
	noProgress   bool
	invalidCount int
}

func (r *loadReader) Read(destination []byte) (count int, err error) {
	r.reads++
	r.largest = max(r.largest, len(destination))
	if r.cancel != nil {
		r.cancel()
	}

	if r.noProgress {
		return 0, nil
	}

	if r.invalidCount != 0 {
		return r.invalidCount, nil
	}

	if r.chunk > 0 {
		destination = destination[:min(len(destination), r.chunk)]
	}

	count = copy(destination, r.data)
	r.data = r.data[count:]
	r.consumed += count
	if len(r.data) != 0 {
		return count, nil
	}

	if r.failure != nil {
		return count, r.failure
	}

	return count, io.EOF
}

func (r *loadReader) Close() (err error) { r.closed = true; return nil }

func TestLoadReader(t *testing.T) {
	r := loadReader{data: []byte{1, 2, 3}, chunk: 2}
	buffer := make([]byte, 4)
	if count, err := r.Read(buffer); count != 2 || err != nil || r.consumed != 2 {
		t.Fatalf("chunk: %d, %v", count, err)
	}

	if count, err := r.Read(buffer); count != 1 || err != io.EOF || r.reads != 2 {
		t.Fatalf("last chunk: %d, %v", count, err)
	}

	if err := r.Close(); err != nil || !r.closed {
		t.Fatalf("close: %v", err)
	}
}
