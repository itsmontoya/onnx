package onnx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"runtime"
	"strings"
	"sync"
)

// NewDirectoryResolver confines external reads to an explicitly chosen directory.
// It uses os.Root's race-safe opens and is unsupported on js and plan9. The
// directory may itself be a symlink. Mounts and hard links are not isolated.
func NewDirectoryResolver(directory string) (out *DirectoryResolver, err error) {
	var r DirectoryResolver
	if directory == "" {
		return nil, ErrInvalidArgument
	}

	if runtime.GOOS == "js" || runtime.GOOS == "plan9" {
		return nil, fmt.Errorf("%w: confined directory resolver on %s; supply a resolver", ErrUnsupported, runtime.GOOS)
	}

	if r.root, err = os.OpenRoot(directory); err != nil {
		return nil, fmt.Errorf("open external directory %q: %w", directory, err)
	}

	return &r, nil
}

// DirectoryResolver opens independent bounded readers beneath a directory handle.
// Locations use relative POSIX paths. Absolute paths, traversal components and
// backslashes are rejected; os.Root rejects symlink escapes during the open.
// Resolve and Close are safe concurrently. Close is idempotent and leaves already
// returned readers usable; callers must close those readers independently.
// A DirectoryResolver must not be copied after first use; its zero value is invalid.
type DirectoryResolver struct {
	mu     sync.RWMutex
	root   *os.Root
	closed bool
}

// Resolve opens a regular file range without sharing reader offsets.
// Offset plus length must fit both int64 and the opened file's current size.
func (r *DirectoryResolver) Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error) {
	if r == nil || ctx == nil {
		return nil, ErrInvalidArgument
	}

	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("resolve external %q: %w", location, err)
	}

	if location == "" || path.IsAbs(location) || strings.ContainsAny(location, "\\:\x00") || offset < 0 || length < 0 || offset > math.MaxInt64-length {
		return nil, fmt.Errorf("resolve external %q: %w: location or range", location, ErrInvalidArgument)
	}

	for _, component := range strings.Split(location, "/") {
		if component == ".." {
			return nil, fmt.Errorf("resolve external %q: %w: traversal", location, ErrInvalidArgument)
		}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.root == nil {
		return nil, ErrInvalidArgument
	}

	if r.closed {
		return nil, ErrClosed
	}

	var file *os.File
	if file, err = r.root.Open(location); err != nil {
		return nil, fmt.Errorf("open external %q: %w", location, err)
	}

	defer func() {
		if source == nil {
			if closeErr := file.Close(); closeErr != nil {
				err = errors.Join(err, fmt.Errorf("close external %q: %w", location, closeErr))
			}
		}
	}()
	var info os.FileInfo
	if info, err = file.Stat(); err != nil {
		return nil, fmt.Errorf("stat external %q: %w", location, err)
	}

	if !info.Mode().IsRegular() || offset > info.Size() || length > info.Size()-offset {
		return nil, fmt.Errorf("resolve external %q: %w: regular file range", location, ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("resolve external %q: %w", location, err)
	}

	return newDirectorySource(ctx, file, offset, length), nil
}

// Close releases the directory handle without closing previously returned readers.
func (r *DirectoryResolver) Close() (err error) {
	if r == nil {
		return ErrInvalidArgument
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.root == nil {
		return ErrInvalidArgument
	}

	if r.closed {
		return nil
	}

	r.closed = true
	if err = r.root.Close(); err != nil {
		return fmt.Errorf("close external directory: %w", err)
	}

	return nil
}
