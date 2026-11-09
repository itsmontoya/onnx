package onnx

import (
	"context"
	"fmt"
	"io"
	"os"
)

func newDirectorySource(ctx context.Context, file *os.File, offset, length int64) (out *directorySource) {
	var s directorySource
	s.ctx, s.file, s.reader = ctx, file, io.NewSectionReader(file, offset, length)
	return &s
}

type directorySource struct {
	ctx    context.Context
	file   *os.File
	reader *io.SectionReader
}

func (s *directorySource) Read(destination []byte) (count int, err error) {
	if err = s.ctx.Err(); err != nil {
		return 0, fmt.Errorf("read external file: %w", err)
	}

	if count, err = s.reader.Read(destination); err != nil && err != io.EOF {
		return count, fmt.Errorf("read external file: %w", err)
	}

	return count, err
}

func (s *directorySource) Close() (err error) {
	if err = s.file.Close(); err != nil {
		return fmt.Errorf("close external file: %w", err)
	}

	return nil
}
