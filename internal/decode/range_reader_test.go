package decode

import (
	"io"
)

type rangeReader struct {
	data       []byte
	failure    error
	closeErr   error
	cancel     func()
	closed     int
	reads      int
	largest    int
	consumed   int
	step       int
	noProgress bool
	badCount   bool
}

func (r *rangeReader) Read(destination []byte) (count int, err error) {
	r.reads++
	r.largest = max(r.largest, len(destination))
	if r.cancel != nil {
		r.cancel()
	}

	if r.badCount {
		return len(destination) + 1, nil
	}

	if r.noProgress {
		return 0, nil
	}

	if r.step > 0 {
		destination = destination[:min(len(destination), r.step)]
	}

	count = copy(destination, r.data)
	r.data = r.data[count:]
	r.consumed += count
	if r.failure != nil {
		return count, r.failure
	}

	if len(r.data) == 0 {
		return count, io.EOF
	}

	return count, nil
}

func (r *rangeReader) Close() (err error) {
	r.closed++
	for index := range r.data {
		r.data[index] = 0
	}
	return r.closeErr
}
