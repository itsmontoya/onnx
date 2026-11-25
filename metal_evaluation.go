package onnx

import (
	"context"
	"fmt"
	"slices"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newMetalEvaluation(ctx context.Context, plan *metalPlan, cache *metalCache, valueCount int) (out *metalEvaluation) {
	var m metalEvaluation
	m.ctx = ctx
	m.plan = plan
	m.cache = cache
	m.values = make([]*gpu.Tensor, valueCount)
	m.shapes = make([][]int64, valueCount)
	m.stats.Active = true
	return &m
}

// metalEvaluation tracks resident Metal tensors and batches consecutive
// GPU-supported operators into one gpu.Scope during a single Session.Run.
type metalEvaluation struct {
	ctx    context.Context
	plan   *metalPlan
	cache  *metalCache
	scope  *gpu.Scope
	values []*gpu.Tensor
	shapes [][]int64
	stats  SessionGPUStats
}

func (m *metalEvaluation) tryStep(e *evaluation, step operators.Step) (handled bool, err error) {
	switch step.Operator {
	case "Identity":
		return m.tryIdentity(e, step)
	case "MatMul":
		return m.tryMatMul(e, step)
	case "Gemm":
		return m.tryGemm(e, step)
	case "Add", "Mul":
		return m.tryElementwise(e, step)
	case "Gather":
		return m.tryGather(e, step)
	default:
		return false, nil
	}
}

func (m *metalEvaluation) tryIdentity(e *evaluation, step operators.Step) (handled bool, err error) {
	inIdx := step.Inputs[0]
	if m.values[inIdx] == nil {
		return false, nil
	}

	outIdx := step.Outputs[0]
	shape := m.shapes[inIdx]
	if err = e.checkValueShape(e.plan.values[outIdx], Float32, shape, ErrInvalidModel); err != nil {
		return true, operationError("run", err, e.plan.values[outIdx].Name)
	}

	m.values[outIdx] = m.values[inIdx]
	m.shapes[outIdx] = shape
	e.storage.values[outIdx] = e.storage.values[inIdx]
	m.stats.DeviceOps++
	return true, nil
}

func (m *metalEvaluation) tryMatMul(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		leftIdx    = step.Inputs[0]
		rightIdx   = step.Inputs[1]
		leftType   DataType
		rightType  DataType
		leftShape  []int64
		rightShape []int64
	)

	leftType, leftShape = m.valueMeta(e, leftIdx)
	rightType, rightShape = m.valueMeta(e, rightIdx)
	if leftType != Float32 || rightType != Float32 {
		return false, nil
	}

	if len(leftShape) != 2 || len(rightShape) != 2 || !positiveShape(leftShape) || !positiveShape(rightShape) || leftShape[1] != rightShape[0] {
		return false, nil
	}

	if m.isReducedPrecision(leftIdx) {
		return false, nil
	}

	outShape := []int64{leftShape[0], rightShape[1]}
	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		left  *gpu.Tensor
		right *gpu.Tensor
		dst   *gpu.Tensor
	)

	if left, err = m.ensureDeviceInput(e, step.Index, 0, leftIdx); err != nil {
		return true, err
	}

	if right, err = m.ensureDeviceInput(e, step.Index, 1, rightIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal MatMul destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.MatMul(left, right, dst); err != nil {
		return true, fmt.Errorf("encode Metal MatMul: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryGemm(e *evaluation, step operators.Step) (handled bool, err error) {
	if step.Gemm.TransposeA || step.Gemm.Alpha != 1 {
		return false, nil
	}

	var (
		leftIdx    = step.Inputs[0]
		rightIdx   = step.Inputs[1]
		hasBias    = len(step.Inputs) > 2 && step.Inputs[2] >= 0
		leftType   DataType
		rightType  DataType
		leftShape  []int64
		rightShape []int64
	)

	leftType, leftShape = m.valueMeta(e, leftIdx)
	rightType, rightShape = m.valueMeta(e, rightIdx)
	if leftType != Float32 || rightType != Float32 {
		return false, nil
	}

	if len(leftShape) != 2 || len(rightShape) != 2 || !positiveShape(leftShape) || !positiveShape(rightShape) {
		return false, nil
	}

	var (
		rows = leftShape[0]
		kDim = leftShape[1]
		cols int64
	)

	if step.Gemm.TransposeB {
		if rightShape[1] != kDim {
			return false, nil
		}

		cols = rightShape[0]
	} else {
		if rightShape[0] != kDim {
			return false, nil
		}

		cols = rightShape[1]
	}

	outShape := []int64{rows, cols}
	var (
		biasIdx  = -1
		biasMode int
	)

	if hasBias {
		biasIdx = step.Inputs[2]
		biasType, biasShape := m.valueMeta(e, biasIdx)
		if biasType != Float32 {
			return false, nil
		}

		if err = tensor.ValidateBroadcastTo(biasShape, outShape); err != nil {
			return false, nil
		}

		if step.Gemm.Beta != 0 {
			if step.Gemm.Beta != 1 {
				return false, nil
			}

			switch {
			case slices.Equal(biasShape, outShape):
				biasMode = 1
			case len(biasShape) == 1 && biasShape[0] == cols:
				biasMode = 2
			case len(biasShape) == 2 && biasShape[0] == 1 && biasShape[1] == cols:
				biasMode = 2
			default:
				return false, nil
			}
		}
	}

	if m.isReducedPrecision(leftIdx) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		left  *gpu.Tensor
		right *gpu.Tensor
		dst   *gpu.Tensor
	)

	if left, err = m.ensureDeviceInput(e, step.Index, 0, leftIdx); err != nil {
		return true, err
	}

	if right, err = m.ensureDeviceInput(e, step.Index, 1, rightIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal Gemm destination: %w", err)
	}

	prodDst := dst
	if biasMode != 0 {
		if prodDst, err = m.cache.acquire(step.Index, 1, outShape); err != nil {
			return true, fmt.Errorf("allocate Metal Gemm product buffer: %w", err)
		}
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if step.Gemm.TransposeB {
		err = m.scope.MatMulRightTranspose(left, right, prodDst)
	} else {
		err = m.scope.MatMul(left, right, prodDst)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal Gemm projection: %w", err)
	}

	if biasMode == 1 {
		var bias *gpu.Tensor
		if bias, err = m.ensureDeviceInput(e, step.Index, 2, biasIdx); err != nil {
			return true, err
		}

		if err = m.scope.Add(prodDst, bias, dst); err != nil {
			return true, fmt.Errorf("encode Metal Gemm bias add: %w", err)
		}
	} else if biasMode == 2 {
		if err = m.encodeRowBiasBinary(e, step.Index, "Add", prodDst, biasIdx, rows, cols, dst); err != nil {
			return true, err
		}
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryElementwise(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		leftIdx    = step.Inputs[0]
		rightIdx   = step.Inputs[1]
		leftType   DataType
		rightType  DataType
		leftShape  []int64
		rightShape []int64
	)

	leftType, leftShape = m.valueMeta(e, leftIdx)
	rightType, rightShape = m.valueMeta(e, rightIdx)
	if leftType != Float32 || rightType != Float32 || !positiveShape(leftShape) || !positiveShape(rightShape) {
		return false, nil
	}

	sameShape := slices.Equal(leftShape, rightShape)
	rowBias := !sameShape && len(leftShape) == 2 &&
		((len(rightShape) == 1 && rightShape[0] == leftShape[1]) ||
			(len(rightShape) == 2 && rightShape[0] == 1 && rightShape[1] == leftShape[1]))
	if !sameShape && !rowBias {
		return false, nil
	}

	if m.isReducedPrecision(leftIdx) || (sameShape && m.isReducedPrecision(rightIdx)) {
		return false, nil
	}

	if err = m.validateAndReserve(e, leftShape); err != nil {
		return true, err
	}

	var (
		left *gpu.Tensor
		dst  *gpu.Tensor
	)

	if left, err = m.ensureDeviceInput(e, step.Index, 0, leftIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, leftShape); err != nil {
		return true, fmt.Errorf("allocate Metal %s destination: %w", step.Operator, err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if sameShape {
		var right *gpu.Tensor
		if right, err = m.ensureDeviceInput(e, step.Index, 1, rightIdx); err != nil {
			return true, err
		}

		if step.Operator == "Add" {
			err = m.scope.Add(left, right, dst)
		} else {
			err = m.scope.Mul(left, right, dst)
		}

		if err != nil {
			return true, fmt.Errorf("encode Metal %s: %w", step.Operator, err)
		}
	} else {
		if err = m.encodeRowBiasBinary(e, step.Index, step.Operator, left, rightIdx, leftShape[0], leftShape[1], dst); err != nil {
			return true, err
		}
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, leftShape)
}

func (m *metalEvaluation) tryGather(e *evaluation, step operators.Step) (handled bool, err error) {
	if step.Axis != 0 {
		return false, nil
	}

	var (
		dataIdx      = step.Inputs[0]
		indicesIdx   = step.Inputs[1]
		dataType     DataType
		indicesType  DataType
		dataShape    []int64
		indicesShape []int64
	)

	dataType, dataShape = m.valueMeta(e, dataIdx)
	indicesType, indicesShape = m.valueMeta(e, indicesIdx)
	if dataType != Float32 || indicesType != Int64 {
		return false, nil
	}

	if len(dataShape) != 2 || len(indicesShape) != 1 || !positiveShape(dataShape) || !positiveShape(indicesShape) {
		return false, nil
	}

	var rawRows []int64
	if rawRows, err = e.storage.values[indicesIdx].Int64Values(); err != nil {
		return true, err
	}

	rows := make([]int64, len(rawRows))
	rowCount := dataShape[0]
	for index, row := range rawRows {
		if row < 0 {
			row += rowCount
		}

		if row < 0 || row >= rowCount {
			return true, fmt.Errorf("%w: Gather index %d out of bounds for axis length %d", ErrInvalidModel, rawRows[index], rowCount)
		}

		rows[index] = row
	}

	outShape := []int64{int64(len(rows)), dataShape[1]}
	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		data *gpu.Tensor
		dst  *gpu.Tensor
	)

	if data, err = m.ensureDeviceInput(e, step.Index, 0, dataIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal Gather destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.GatherRows(data, rows, dst); err != nil {
		return true, fmt.Errorf("encode Metal GatherRows: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) encodeRowBiasBinary(
	e *evaluation,
	stepIndex int,
	operator string,
	left *gpu.Tensor,
	biasIdx int,
	rows int64,
	cols int64,
	dst *gpu.Tensor,
) (err error) {
	var bias *gpu.Tensor
	if bias, err = m.ensureDeviceRowBias(e, stepIndex, 2, biasIdx, cols); err != nil {
		return err
	}

	operand := bias
	if rows > 1 {
		var bcast *gpu.Tensor
		if bcast, err = m.cache.acquire(stepIndex, 2, []int64{rows, cols}); err != nil {
			return fmt.Errorf("allocate Metal bias broadcast buffer: %w", err)
		}

		zeroRows := make([]int64, rows)
		if err = m.scope.GatherRows(bias, zeroRows, bcast); err != nil {
			return fmt.Errorf("encode Metal bias broadcast: %w", err)
		}

		operand = bcast
	}

	if operator == "Add" {
		err = m.scope.Add(left, operand, dst)
	} else {
		err = m.scope.Mul(left, operand, dst)
	}

	if err != nil {
		return fmt.Errorf("encode Metal %s row bias: %w", operator, err)
	}

	return nil
}

func (m *metalEvaluation) publishGPUValue(e *evaluation, outIdx int, dst *gpu.Tensor, shape []int64) (err error) {
	if err = e.checkValueShape(e.plan.values[outIdx], Float32, shape, ErrInvalidModel); err != nil {
		return operationError("run", err, e.plan.values[outIdx].Name)
	}

	m.values[outIdx] = dst
	m.shapes[outIdx] = shape
	e.storage.values[outIdx] = nil
	m.stats.DeviceOps++
	return nil
}

func (m *metalEvaluation) validateAndReserve(e *evaluation, shape []int64) (err error) {
	if len(shape) > e.plan.limits.MaxRank {
		return fmt.Errorf("%w: numerical output rank", ErrLimit)
	}

	for _, size := range shape {
		if size > e.plan.limits.MaxDimension {
			return fmt.Errorf("%w: numerical output dimension", ErrLimit)
		}
	}

	var bytes int64
	if _, bytes, err = tensor.ValidateShape(shape, Float32); err != nil {
		return fmt.Errorf("%w: numerical output: %w", ErrLimit, err)
	}

	if err = e.storage.reserve(bytes, 0); err != nil {
		return err
	}

	return nil
}

func (m *metalEvaluation) ensureScope() (err error) {
	if m.scope != nil {
		return nil
	}

	if m.scope, err = m.plan.device.NewScope(); err != nil {
		return fmt.Errorf("create Metal scope: %w", err)
	}

	return nil
}

func (m *metalEvaluation) ensureDeviceInput(e *evaluation, stepIndex, slot, valueIndex int) (out *gpu.Tensor, err error) {
	if m.values[valueIndex] != nil {
		return m.values[valueIndex], nil
	}

	if weight, ok := m.plan.weights[valueIndex]; ok {
		return weight, nil
	}

	host := e.storage.values[valueIndex]
	if host == nil {
		return nil, fmt.Errorf("%w: missing value %d for Metal upload", ErrInvalidModel, valueIndex)
	}

	var values []float32
	if values, err = host.Float32Values(); err != nil {
		return nil, fmt.Errorf("read host input for Metal upload: %w", err)
	}

	shape := host.Shape()
	if out, err = m.cache.acquire(stepIndex, 10+slot, shape); err != nil {
		return nil, fmt.Errorf("allocate Metal input buffer: %w", err)
	}

	if err = out.Upload(values); err != nil {
		return nil, fmt.Errorf("upload host input to Metal: %w", err)
	}

	m.values[valueIndex] = out
	m.shapes[valueIndex] = shape
	m.stats.HostUploads++
	return out, nil
}

func (m *metalEvaluation) ensureDeviceRowBias(e *evaluation, stepIndex, slot, valueIndex int, cols int64) (out *gpu.Tensor, err error) {
	if bias, ok := m.plan.rowBiases[valueIndex]; ok {
		return bias, nil
	}

	if m.values[valueIndex] != nil && slices.Equal(m.shapes[valueIndex], []int64{1, cols}) {
		return m.values[valueIndex], nil
	}

	if weight, ok := m.plan.weights[valueIndex]; ok && slices.Equal(weight.Shape(), []int64{1, cols}) && weight.DType() == gpu.Float32 {
		return weight, nil
	}

	var host *tensor.Tensor
	if host, err = m.materialize(e, valueIndex); err != nil {
		return nil, err
	}

	var values []float32
	if values, err = host.Float32Values(); err != nil {
		return nil, fmt.Errorf("read bias for Metal upload: %w", err)
	}

	if out, err = m.cache.acquire(stepIndex, 20+slot, []int64{1, cols}); err != nil {
		return nil, fmt.Errorf("allocate Metal bias buffer: %w", err)
	}

	if err = out.Upload(values); err != nil {
		return nil, fmt.Errorf("upload bias to Metal: %w", err)
	}

	m.stats.HostUploads++
	return out, nil
}

func (m *metalEvaluation) materialize(e *evaluation, valueIndex int) (out *tensor.Tensor, err error) {
	if valueIndex < 0 {
		return nil, nil
	}

	if e.storage.values[valueIndex] != nil {
		return e.storage.values[valueIndex], nil
	}

	resident := m.values[valueIndex]
	if resident == nil {
		return nil, nil
	}

	if err = m.flush(); err != nil {
		return nil, err
	}

	shape := m.shapes[valueIndex]
	var elements int64
	if elements, _, err = tensor.ValidateShape(shape, Float32); err != nil {
		return nil, err
	}

	buf := make([]float32, elements)
	if err = resident.DownloadInto(buf); err != nil {
		return nil, fmt.Errorf("download Metal tensor: %w", err)
	}

	if out, err = tensor.FromFloat32(shape, buf); err != nil {
		return nil, fmt.Errorf("construct host tensor from Metal download: %w", err)
	}

	e.storage.values[valueIndex] = out
	m.stats.HostDownloads++
	return out, nil
}

func (m *metalEvaluation) materializeInputs(e *evaluation, step operators.Step) (err error) {
	for _, valueIndex := range step.Inputs {
		if valueIndex < 0 || e.storage.values[valueIndex] != nil || m.values[valueIndex] == nil {
			continue
		}

		if _, err = m.materialize(e, valueIndex); err != nil {
			return err
		}
	}

	return nil
}

func (m *metalEvaluation) flush() (err error) {
	if m.scope == nil {
		return nil
	}

	scope := m.scope
	m.scope = nil
	if err = scope.Submit(m.ctx); err != nil {
		_ = scope.Close()
		return fmt.Errorf("submit Metal scope: %w", err)
	}

	m.stats.ScopeSubmits++
	if err = scope.Wait(m.ctx); err != nil {
		_ = scope.Close()
		return fmt.Errorf("wait Metal scope: %w", err)
	}

	if err = scope.Close(); err != nil {
		return fmt.Errorf("close Metal scope: %w", err)
	}

	return nil
}

func (m *metalEvaluation) valueMeta(e *evaluation, valueIndex int) (dataType DataType, shape []int64) {
	if m.values[valueIndex] != nil {
		return Float32, m.shapes[valueIndex]
	}

	if e.storage.values[valueIndex] != nil {
		return e.storage.values[valueIndex].DataType(), e.storage.values[valueIndex].Shape()
	}

	return Invalid, nil
}

func (m *metalEvaluation) isReducedPrecision(valueIndex int) (reduced bool) {
	if m.values[valueIndex] != nil {
		return m.values[valueIndex].DType() != gpu.Float32
	}

	if weight, ok := m.plan.weights[valueIndex]; ok {
		return weight.DType() != gpu.Float32
	}

	return false
}

func (m *metalEvaluation) shapeOf(valueIndex int) (shape []int64) {
	if m == nil || valueIndex < 0 || valueIndex >= len(m.shapes) {
		return nil
	}

	return m.shapes[valueIndex]
}

func (m *metalEvaluation) close() (err error) {
	if m == nil {
		return nil
	}

	if m.scope != nil {
		err = m.scope.Close()
		m.scope = nil
	}

	return err
}
