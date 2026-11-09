package onnx

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func directoryFixture(t *testing.T) (directory string, resolver *DirectoryResolver) {
	t.Helper()
	if runtime.GOOS == "js" || runtime.GOOS == "plan9" {
		t.Skip("os.Root confinement is unavailable")
	}

	directory = t.TempDir()
	var err error
	if err = os.WriteFile(filepath.Join(directory, "weights.bin"), []byte("prefix|first|second|tail"), 0600); err != nil {
		t.Fatal(err)
	}

	if resolver, err = NewDirectoryResolver(directory); err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if closeErr := resolver.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return directory, resolver
}

func TestDirectoryResolverRangesAndLifetime(t *testing.T) {
	_, resolver := directoryFixture(t)
	ctx := context.Background()
	first, err := resolver.Resolve(ctx, "weights.bin", 7, 5)
	if err != nil {
		t.Fatal(err)
	}

	defer first.Close()
	second, err := resolver.Resolve(ctx, "weights.bin", 13, 6)
	if err != nil {
		t.Fatal(err)
	}

	defer second.Close()
	if err = resolver.Close(); err != nil {
		t.Fatal(err)
	}

	if err = resolver.Close(); err != nil {
		t.Fatal(err)
	}

	for index, source := range []io.Reader{second, first} {
		data, readErr := io.ReadAll(source)
		if readErr != nil || string(data) != []string{"second", "first"}[index] {
			t.Fatalf("independent bounded reader: %q, %v", data, readErr)
		}
	}
	if source, resolveErr := resolver.Resolve(ctx, "weights.bin", 0, 1); source != nil || !errors.Is(resolveErr, ErrClosed) {
		t.Fatalf("closed resolver: %v, %v", source, resolveErr)
	}
}

func TestDirectoryResolverValidation(t *testing.T) {
	directory, resolver := directoryFixture(t)
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	var zero DirectoryResolver
	tests := []struct {
		name     string
		resolver *DirectoryResolver
		ctx      context.Context
		location string
		offset   int64
		length   int64
		want     error
	}{
		{"nil", nil, context.Background(), "weights.bin", 0, 1, ErrInvalidArgument},
		{"zero", &zero, context.Background(), "weights.bin", 0, 1, ErrInvalidArgument},
		{"nil context", resolver, nil, "weights.bin", 0, 1, ErrInvalidArgument},
		{"cancelled", resolver, cancelled, "weights.bin", 0, 1, context.Canceled},
		{"empty", resolver, context.Background(), "", 0, 1, ErrInvalidArgument},
		{"absolute", resolver, context.Background(), filepath.Join(directory, "weights.bin"), 0, 1, ErrInvalidArgument},
		{"traversal", resolver, context.Background(), "../weights.bin", 0, 1, ErrInvalidArgument},
		{"inner traversal", resolver, context.Background(), "a/../weights.bin", 0, 1, ErrInvalidArgument},
		{"backslash", resolver, context.Background(), "a\\weights.bin", 0, 1, ErrInvalidArgument},
		{"drive", resolver, context.Background(), "C:/weights.bin", 0, 1, ErrInvalidArgument},
		{"url", resolver, context.Background(), "https://example.com/weights.bin", 0, 1, ErrInvalidArgument},
		{"nul", resolver, context.Background(), "weights\x00.bin", 0, 1, ErrInvalidArgument},
		{"negative offset", resolver, context.Background(), "weights.bin", -1, 1, ErrInvalidArgument},
		{"negative length", resolver, context.Background(), "weights.bin", 0, -1, ErrInvalidArgument},
		{"overflow", resolver, context.Background(), "weights.bin", math.MaxInt64, 1, ErrInvalidArgument},
		{"past end", resolver, context.Background(), "weights.bin", 25, 0, ErrInvalidArgument},
		{"range past end", resolver, context.Background(), "weights.bin", 23, 2, ErrInvalidArgument},
		{"directory", resolver, context.Background(), ".", 0, 0, ErrInvalidArgument},
		{"missing", resolver, context.Background(), "missing", 0, 1, os.ErrNotExist},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			source, err := testcase.resolver.Resolve(testcase.ctx, testcase.location, testcase.offset, testcase.length)
			if source != nil || !errors.Is(err, testcase.want) {
				t.Fatalf("resolve: %v, %v", source, err)
			}
		})
	}
	if err := zero.Close(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}

	if err := (*DirectoryResolver)(nil).Close(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatal(err)
	}

	if value, err := NewDirectoryResolver(""); value != nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("empty root: %v, %v", value, err)
	}

	if value, err := NewDirectoryResolver(filepath.Join(directory, "missing")); value != nil || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing root: %v, %v", value, err)
	}

	for _, offset := range []int64{0, 24} {
		source, err := resolver.Resolve(context.Background(), "weights.bin", offset, 0)
		if err != nil {
			t.Fatal(err)
		}

		data, readErr := io.ReadAll(source)
		closeErr := source.Close()
		if len(data) != 0 || readErr != nil || closeErr != nil {
			t.Fatalf("zero range: %v, %v, %v", data, readErr, closeErr)
		}
	}
}

func TestDirectoryResolverSymlinks(t *testing.T) {
	directory, resolver := directoryFixture(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("OUTSIDE"), 0600); err != nil {
		t.Fatal(err)
	}

	links := map[string]string{
		"inside":      "weights.bin",
		"escape":      filepath.Join(outside, "secret"),
		"parent":      "../" + filepath.Base(outside) + "/secret",
		"outside-dir": outside,
	}
	for name, target := range links {
		if err := os.Symlink(target, filepath.Join(directory, name)); err != nil {
			t.Skipf("symlink creation unavailable: %v", err)
		}
	}
	for _, name := range []string{"escape", "parent", "outside-dir/secret"} {
		source, err := resolver.Resolve(context.Background(), name, 0, 7)
		if source != nil || err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("escaped: %v, %v", source, err)
		}
	}
	source, err := resolver.Resolve(context.Background(), "inside", 7, 5)
	if err != nil {
		t.Fatal(err)
	}

	data, readErr := io.ReadAll(source)
	closeErr := source.Close()
	if string(data) != "first" || readErr != nil || closeErr != nil {
		t.Fatalf("inside link: %q, %v, %v", data, readErr, closeErr)
	}
}

func TestDirectoryResolverRacingReplacement(t *testing.T) {
	directory, resolver := directoryFixture(t)
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("OUTSIDE"), 0600); err != nil {
		t.Fatal(err)
	}

	target := filepath.Join(directory, "swapped")
	if err := os.Symlink("weights.bin", target); err != nil {
		t.Skipf("symlink creation unavailable: %v", err)
	}

	stop := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		var err error
		defer func() { done <- err }()
		for {
			for _, destination := range []string{"weights.bin", filepath.Join(outside, "secret")} {
				select {
				case <-stop:
					return
				default:
				}
				pending := filepath.Join(directory, "pending-link")
				if err = os.Symlink(destination, pending); err != nil {
					return
				}

				if err = os.Rename(pending, target); err != nil {
					return
				}
			}
		}
	}()
	defer func() {
		close(stop)
		if err := <-done; err != nil {
			t.Error(err)
		}
	}()
	for range 1000 {
		source, err := resolver.Resolve(context.Background(), "swapped", 0, 7)
		if err != nil {
			continue
		}

		data, readErr := io.ReadAll(source)
		closeErr := source.Close()
		if !bytes.Equal(data, []byte("prefix|")) || readErr != nil || closeErr != nil {
			t.Fatalf("racing escape: %q, %v, %v", data, readErr, closeErr)
		}
	}
}

func TestDirectoryResolverConcurrentReadsAndClose(t *testing.T) {
	_, resolver := directoryFixture(t)
	var workers sync.WaitGroup
	for worker := range 8 {
		workers.Add(1)
		go func(position int) {
			defer workers.Done()
			for range 30 {
				offset, length := int64(7), int64(5)
				want := "first"
				if position%2 != 0 {
					offset, length, want = 13, 6, "second"
				}

				source, err := resolver.Resolve(context.Background(), "weights.bin", offset, length)
				if errors.Is(err, ErrClosed) {
					return
				}

				if err != nil {
					t.Error(err)
					return
				}

				data, readErr := io.ReadAll(source)
				closeErr := source.Close()
				if string(data) != want || readErr != nil || closeErr != nil {
					t.Errorf("concurrent range: %q, %v, %v", data, readErr, closeErr)
					return
				}
			}
		}(worker)
	}
	if err := resolver.Close(); err != nil {
		t.Fatal(err)
	}

	workers.Wait()
}
