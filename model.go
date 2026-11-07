package onnx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"github.com/itsmontoya/onnx/internal/operators"
)

// Load reads, validates and compiles a bounded model without closing source.
// Readers and contexts must be nonnil; invalid options fail before reading.
func Load(ctx context.Context, source io.Reader, options Options) (out *Model, err error) {
	if options, err = validateLoad(ctx, options); err != nil {
		return nil, operationError("load", err, "")
	}

	if source == nil {
		return nil, ErrInvalidArgument
	}

	var (
		wire         *onnxpb.ModelProto
		encodedBytes int64
	)

	if wire, encodedBytes, err = decode.ReadModelWithSize(ctx, source, options.Limits.wire()); err != nil {
		return nil, operationError("load", err, "")
	}

	return loadModel(ctx, wire, encodedBytes, options.Limits)
}

// LoadBytes borrows data only for the call and publishes an owned immutable plan.
// The caller must not mutate data during loading.
func LoadBytes(ctx context.Context, data []byte, options Options) (out *Model, err error) {
	if options, err = validateLoad(ctx, options); err != nil {
		return nil, operationError("load", err, "")
	}

	var wire *onnxpb.ModelProto
	if wire, err = decode.ParseModel(ctx, data, options.Limits.wire()); err != nil {
		return nil, operationError("load", err, "")
	}

	return loadModel(ctx, wire, int64(len(data)), options.Limits)
}

// LoadFile owns and closes only the file it opens. It never guesses a directory
// for external data; external tensors and resolver options remain unsupported.
func LoadFile(ctx context.Context, path string, options Options) (out *Model, err error) {
	if options, err = validateLoad(ctx, options); err != nil {
		return nil, operationError("load", err, "")
	}

	if path == "" {
		return nil, ErrInvalidArgument
	}

	var source *os.File
	if source, err = os.Open(path); err != nil {
		return nil, operationError("load", fmt.Errorf("open model %q: %w", path, err), "")
	}

	out, err = Load(ctx, source, options)
	if closeErr := source.Close(); closeErr != nil {
		if out != nil {
			_ = out.Close()
		}

		return nil, operationError("load", errors.Join(err, fmt.Errorf("close model %q: %w", path, closeErr)), "")
	}

	return out, err
}

func newModel(plan *executionPlan) (out *Model) {
	var m Model
	m.plan, m.metadata, m.initialized = plan, plan.metadata, true
	return &m
}

// Model shares an immutable plan across sessions. Close prevents new sessions;
// existing sessions retain the plan and Metadata remains available afterward.
// A Model must not be copied after first use. Its zero value is invalid.
type Model struct {
	mu          sync.Mutex
	plan        *executionPlan
	metadata    Metadata
	initialized bool
	closed      bool
}

// Metadata returns independent input/output records, including after Close.
// Nil and zero models return zero metadata.
func (m *Model) Metadata() (metadata Metadata) {
	if m == nil || !m.initialized {
		return metadata
	}

	return m.metadata.clone()
}

// NewSession reserves one of the plan's finite session slots. Creation and
// Model.Close are linearized; a successfully created session remains usable.
func (m *Model) NewSession() (out *Session, err error) {
	if m == nil {
		return nil, ErrInvalidArgument
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.initialized {
		return nil, ErrInvalidArgument
	}

	if m.closed {
		return nil, ErrClosed
	}

	if err = m.plan.acquire(); err != nil {
		return nil, err
	}

	return newSession(m.plan), nil
}

// Close is idempotent and releases this model's plan reference.
func (m *Model) Close() (err error) {
	if m == nil {
		return ErrInvalidArgument
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.initialized {
		return ErrInvalidArgument
	}

	m.closed, m.plan = true, nil
	return nil
}

func validateLoad(ctx context.Context, options Options) (out Options, err error) {
	if ctx == nil {
		return out, ErrInvalidArgument
	}

	if out, err = options.normalized(); err != nil {
		return out, err
	}

	if err = ctx.Err(); err != nil {
		return out, err
	}

	return out, nil
}

func loadModel(ctx context.Context, wire *onnxpb.ModelProto, encodedBytes int64, limits Limits) (out *Model, err error) {
	var (
		source   *graph.Graph
		compiled *operators.Plan
		plan     *executionPlan
	)

	if source, err = graph.Build(ctx, wire, encodedBytes, limits.graph()); err != nil {
		return nil, operationError("load", err, "")
	}

	if compiled, err = compileGraph(ctx, source, limits.compilation()); err != nil {
		return nil, operationError("load", err, "")
	}

	if plan, err = newExecutionPlan(ctx, compiled, limits); err != nil {
		return nil, operationError("load", err, "")
	}

	if err = ctx.Err(); err != nil {
		return nil, operationError("load", err, "")
	}

	return newModel(plan), nil
}
