package decode

import (
	"context"
	"io"
)

type testReader struct {
	data           []byte
	failure        error
	chunk          int
	readBytes      int
	reads          int
	largestRequest int
	closed         bool
	noProgress     bool
	invalidCount   int
	cancel         context.CancelFunc
}

func (r *testReader) Read(destination []byte) (count int, err error) {
	r.reads++
	r.largestRequest = max(r.largestRequest, len(destination))
	if r.cancel != nil {
		r.cancel()
	}

	if r.invalidCount != 0 {
		return r.invalidCount, nil
	}

	if r.noProgress {
		return 0, nil
	}

	if r.chunk > 0 {
		destination = destination[:min(len(destination), r.chunk)]
	}

	count = copy(destination, r.data)
	r.data = r.data[count:]
	r.readBytes += count
	if len(r.data) > 0 {
		return count, nil
	}

	if r.failure != nil {
		return count, r.failure
	}

	return count, io.EOF
}

func (r *testReader) Close() (err error) {
	r.closed = true
	return nil
}
