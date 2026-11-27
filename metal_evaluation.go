package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
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
	views  []*gpu.Tensor
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
	case "Add", "Sub", "Mul", "Div":
		return m.tryElementwise(e, step)
	case "Relu", "Sigmoid", "Tanh":
		return m.tryActivation(e, step)
	case "Softmax":
		return m.trySoftmax(e, step)
	case "LayerNormalization":
		return m.tryLayerNorm(e, step)
	case "Transpose":
		return m.tryTranspose(e, step)
	case "Reshape", "Flatten", "Squeeze", "Unsqueeze":
		return m.tryShapeView(e, step)
	case "Gather":
		return m.tryGather(e, step)
	case "Conv":
		return m.tryConv(e, step)
	case "BatchNormalization":
		return m.tryBatchNorm(e, step)
	case "MaxPool", "AveragePool", "GlobalAveragePool":
		return m.tryPool2D(e, step)
	case "Concat":
		return m.tryConcat(e, step)
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

func (m *metalEvaluation) tryActivation(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		inIdx   = step.Inputs[0]
		inType  DataType
		inShape []int64
	)

	inType, inShape = m.valueMeta(e, inIdx)
	if inType != Float32 || !validGPUShape(inShape) || m.isReducedPrecision(inIdx) {
		return false, nil
	}

	if err = m.validateAndReserve(e, inShape); err != nil {
		return true, err
	}

	var (
		input *gpu.Tensor
		dst   *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, inIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, inShape); err != nil {
		return true, fmt.Errorf("allocate Metal %s destination: %w", step.Operator, err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	switch step.Operator {
	case "Relu":
		err = m.scope.ReLU(input, dst)
	case "Sigmoid":
		err = m.scope.Sigmoid(input, dst)
	case "Tanh":
		err = m.scope.Tanh(input, dst)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal %s: %w", step.Operator, err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, inShape)
}

func (m *metalEvaluation) trySoftmax(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		inIdx   = step.Inputs[0]
		inType  DataType
		inShape []int64
		axis    int
	)

	inType, inShape = m.valueMeta(e, inIdx)
	if inType != Float32 || !validGPUShape(inShape) || m.isReducedPrecision(inIdx) {
		return false, nil
	}

	if axis, err = tensor.NormalizeAxis(step.Axis, len(inShape)); err != nil {
		return false, nil
	}

	if err = m.validateAndReserve(e, inShape); err != nil {
		return true, err
	}

	var (
		input *gpu.Tensor
		dst   *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, inIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, inShape); err != nil {
		return true, fmt.Errorf("allocate Metal Softmax destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if axis == len(inShape)-1 {
		err = m.scope.Softmax(input, dst)
	} else {
		err = m.scope.SoftmaxAxis(input, dst, axis)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal Softmax: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, inShape)
}

func (m *metalEvaluation) tryMatMul(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		leftIdx    = step.Inputs[0]
		rightIdx   = step.Inputs[1]
		leftType   DataType
		rightType  DataType
		leftShape  []int64
		rightShape []int64
		outShape   []int64
	)

	leftType, leftShape = m.valueMeta(e, leftIdx)
	rightType, rightShape = m.valueMeta(e, rightIdx)
	if leftType != Float32 || rightType != Float32 {
		return false, nil
	}

	if len(leftShape) < 2 || len(rightShape) < 2 || len(leftShape) > 6 || len(rightShape) > 6 {
		return false, nil
	}

	if !validGPUShape(leftShape) || !validGPUShape(rightShape) || m.isReducedPrecision(leftIdx) {
		return false, nil
	}

	if outShape, err = tensor.MatMulShape(leftShape, rightShape); err != nil || len(outShape) > 6 || !validGPUShape(outShape) {
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
		return true, fmt.Errorf("allocate Metal MatMul destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if len(leftShape) == 2 && len(rightShape) == 2 {
		err = m.scope.MatMul(left, right, dst)
	} else {
		err = m.scope.BatchedMatMul(left, right, dst)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal MatMul: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryGemm(e *evaluation, step operators.Step) (handled bool, err error) {
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

	if len(leftShape) != 2 || len(rightShape) != 2 || !validGPUShape(leftShape) || !validGPUShape(rightShape) {
		return false, nil
	}

	if m.isReducedPrecision(leftIdx) {
		return false, nil
	}

	var (
		rows int64
		kDim int64
		cols int64
	)

	if step.Gemm.TransposeA {
		kDim = leftShape[0]
		rows = leftShape[1]
	} else {
		rows = leftShape[0]
		kDim = leftShape[1]
	}

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
	if !validGPUShape(outShape) {
		return false, nil
	}

	var (
		biasIdx   = -1
		biasShape []int64
		useBias   bool
	)

	if hasBias {
		biasIdx = step.Inputs[2]
		var biasType DataType
		biasType, biasShape = m.valueMeta(e, biasIdx)
		if biasType != Float32 || m.isReducedPrecision(biasIdx) {
			return false, nil
		}

		if len(biasShape) > 0 && !validGPUShape(biasShape) {
			return false, nil
		}

		if err = tensor.ValidateBroadcastTo(biasShape, outShape); err != nil {
			return false, nil
		}

		useBias = step.Gemm.Beta != 0
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

	needAlpha := step.Gemm.Alpha != 1
	prodDst := dst
	if needAlpha || useBias {
		if prodDst, err = m.cache.acquire(step.Index, 1, outShape); err != nil {
			return true, fmt.Errorf("allocate Metal Gemm product buffer: %w", err)
		}
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	switch {
	case !step.Gemm.TransposeA && !step.Gemm.TransposeB:
		err = m.scope.MatMul(left, right, prodDst)
	case !step.Gemm.TransposeA && step.Gemm.TransposeB:
		err = m.scope.MatMulRightTranspose(left, right, prodDst)
	case step.Gemm.TransposeA && !step.Gemm.TransposeB:
		err = m.scope.MatMulLeftTranspose(left, right, prodDst)
	default:
		var leftT *gpu.Tensor
		if leftT, err = m.cache.acquire(step.Index, 3, []int64{rows, kDim}); err != nil {
			return true, fmt.Errorf("allocate Metal Gemm transposed left buffer: %w", err)
		}

		if err = m.scope.Transpose(left, leftT, []int{1, 0}); err != nil {
			return true, fmt.Errorf("encode Metal Gemm left transpose: %w", err)
		}

		err = m.scope.MatMulRightTranspose(leftT, right, prodDst)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal Gemm projection: %w", err)
	}

	current := prodDst
	if needAlpha {
		scaleDst := dst
		if useBias {
			if scaleDst, err = m.cache.acquire(step.Index, 2, outShape); err != nil {
				return true, fmt.Errorf("allocate Metal Gemm alpha buffer: %w", err)
			}
		}

		if err = m.scope.Scale(current, scaleDst, step.Gemm.Alpha); err != nil {
			return true, fmt.Errorf("encode Metal Gemm alpha scale: %w", err)
		}

		current = scaleDst
	}

	if useBias {
		var bias *gpu.Tensor
		if bias, err = m.ensureDeviceInput(e, step.Index, 2, biasIdx); err != nil {
			return true, err
		}

		biasOperand := bias
		if step.Gemm.Beta != 1 {
			var scaledBias *gpu.Tensor
			if scaledBias, err = m.cache.acquire(step.Index, 4, bias.Shape()); err != nil {
				return true, fmt.Errorf("allocate Metal Gemm beta buffer: %w", err)
			}

			if err = m.scope.Scale(bias, scaledBias, step.Gemm.Beta); err != nil {
				return true, fmt.Errorf("encode Metal Gemm beta scale: %w", err)
			}

			biasOperand = scaledBias
		}

		if slices.Equal(biasShape, outShape) {
			err = m.scope.Add(current, biasOperand, dst)
		} else {
			err = m.scope.AddBroadcast(current, biasOperand, dst)
		}

		if err != nil {
			return true, fmt.Errorf("encode Metal Gemm bias add: %w", err)
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
		outShape   []int64
	)

	leftType, leftShape = m.valueMeta(e, leftIdx)
	rightType, rightShape = m.valueMeta(e, rightIdx)
	if leftType != Float32 || rightType != Float32 {
		return false, nil
	}

	if (len(leftShape) > 0 && !validGPUShape(leftShape)) || (len(rightShape) > 0 && !validGPUShape(rightShape)) {
		return false, nil
	}

	if outShape, err = tensor.BroadcastShape(leftShape, rightShape); err != nil || len(outShape) == 0 || len(outShape) > 6 || !validGPUShape(outShape) {
		return false, nil
	}

	if m.isReducedPrecision(leftIdx) || m.isReducedPrecision(rightIdx) {
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
		return true, fmt.Errorf("allocate Metal %s destination: %w", step.Operator, err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	sameShape := slices.Equal(leftShape, rightShape) && len(leftShape) > 0
	if sameShape {
		switch step.Operator {
		case "Add":
			err = m.scope.Add(left, right, dst)
		case "Sub":
			err = m.scope.Sub(left, right, dst)
		case "Mul":
			err = m.scope.Mul(left, right, dst)
		case "Div":
			err = m.scope.Div(left, right, dst)
		}
	} else {
		switch step.Operator {
		case "Add":
			err = m.scope.AddBroadcast(left, right, dst)
		case "Sub":
			err = m.scope.SubBroadcast(left, right, dst)
		case "Mul":
			err = m.scope.MulBroadcast(left, right, dst)
		case "Div":
			err = m.scope.DivBroadcast(left, right, dst)
		}
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal %s: %w", step.Operator, err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryLayerNorm(e *evaluation, step operators.Step) (handled bool, err error) {
	if len(step.Outputs) > 1 && step.Outputs[1] >= 0 {
		return false, nil
	}

	if len(step.Outputs) > 2 && step.Outputs[2] >= 0 {
		return false, nil
	}

	if !(step.Epsilon > 0) || math.IsNaN(float64(step.Epsilon)) || math.IsInf(float64(step.Epsilon), 0) {
		return false, nil
	}

	var (
		xIdx    = step.Inputs[0]
		wIdx    = step.Inputs[1]
		hasBias = len(step.Inputs) > 2 && step.Inputs[2] >= 0
		xType   DataType
		wType   DataType
		xShape  []int64
		wShape  []int64
		axis    int
	)

	xType, xShape = m.valueMeta(e, xIdx)
	wType, wShape = m.valueMeta(e, wIdx)
	if xType != Float32 || wType != Float32 || !validGPUShape(xShape) || m.isReducedPrecision(xIdx) {
		return false, nil
	}

	if axis, err = tensor.NormalizeAxis(step.Axis, len(xShape)); err != nil || axis != len(xShape)-1 {
		return false, nil
	}

	width := xShape[len(xShape)-1]
	if len(wShape) != 1 || wShape[0] != width {
		return false, nil
	}

	bIdx := -1
	if hasBias {
		bIdx = step.Inputs[2]
		bType, bShape := m.valueMeta(e, bIdx)
		if bType != Float32 || len(bShape) != 1 || bShape[0] != width {
			return false, nil
		}
	}

	if err = m.validateAndReserve(e, xShape); err != nil {
		return true, err
	}

	var (
		input  *gpu.Tensor
		weight *gpu.Tensor
		bias   *gpu.Tensor
		dst    *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, xIdx); err != nil {
		return true, err
	}

	if weight, err = m.ensureDeviceInput(e, step.Index, 1, wIdx); err != nil {
		return true, err
	}

	if hasBias {
		if bias, err = m.ensureDeviceInput(e, step.Index, 2, bIdx); err != nil {
			return true, err
		}
	}

	if dst, err = m.cache.acquire(step.Index, 0, xShape); err != nil {
		return true, fmt.Errorf("allocate Metal LayerNormalization destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.LayerNorm(input, weight, bias, dst, step.Epsilon); err != nil {
		return true, fmt.Errorf("encode Metal LayerNormalization: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, xShape)
}

func (m *metalEvaluation) tryTranspose(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		inIdx    = step.Inputs[0]
		inType   DataType
		inShape  []int64
		outShape []int64
	)

	inType, inShape = m.valueMeta(e, inIdx)
	if inType != Float32 || len(inShape) == 0 || len(inShape) > 6 || !validGPUShape(inShape) || m.isReducedPrecision(inIdx) {
		return false, nil
	}

	rawPerm := step.Permutation
	if !step.PermutationSet {
		rawPerm = operators.ReversePermutation(len(inShape))
	}

	if err = operators.ValidatePermutation(rawPerm, len(inShape)); err != nil {
		return false, nil
	}

	if outShape, err = tensor.TransposeShape(inShape, rawPerm); err != nil || !validGPUShape(outShape) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	perm := make([]int, len(rawPerm))
	for i, axis := range rawPerm {
		perm[i] = int(axis)
	}

	var (
		input *gpu.Tensor
		dst   *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, inIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal Transpose destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.Transpose(input, dst, perm); err != nil {
		return true, fmt.Errorf("encode Metal Transpose: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryShapeView(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		inIdx    = step.Inputs[0]
		inType   DataType
		inShape  []int64
		outShape []int64
	)

	inType, inShape = m.valueMeta(e, inIdx)
	if inType != Float32 || !validGPUShape(inShape) {
		return false, nil
	}

	var axes []int64
	if len(step.Inputs) > 1 && step.Inputs[1] >= 0 {
		axesTensor := e.storage.values[step.Inputs[1]]
		if axesTensor == nil || axesTensor.DataType() != Int64 || axesTensor.Rank() != 1 {
			return false, nil
		}

		count := axesTensor.Len()
		if count > int64(e.plan.limits.MaxRank) ||
			(step.Operator == "Squeeze" && count > int64(len(inShape))) ||
			(step.Operator == "Unsqueeze" && count > int64(e.plan.limits.MaxRank-len(inShape))) {
			return false, nil
		}

		if axes, err = axesTensor.Int64Values(); err != nil {
			return false, nil
		}
	}

	switch step.Operator {
	case "Reshape":
		outShape, err = operators.ResolveReshapeMode(inShape, axes, step.AllowZero)
	case "Flatten":
		outShape, err = tensor.FlattenShape(inShape, step.Axis)
	case "Squeeze":
		if len(step.Inputs) == 1 || step.Inputs[1] < 0 {
			for axis, dimension := range inShape {
				if dimension == 1 {
					axes = append(axes, int64(axis))
				}
			}
		}

		outShape, err = tensor.SqueezeShape(inShape, axes)
	case "Unsqueeze":
		outShape, err = tensor.UnsqueezeShape(inShape, axes)
	}

	if err != nil || !validGPUShape(outShape) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		input *gpu.Tensor
		view  *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, inIdx); err != nil {
		return true, err
	}

	if view, err = input.View(outShape); err != nil {
		return true, fmt.Errorf("create Metal %s view: %w", step.Operator, err)
	}

	m.views = append(m.views, view)
	return true, m.publishGPUValue(e, step.Outputs[0], view, outShape)
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

	if len(dataShape) != 2 || len(indicesShape) != 1 || !validGPUShape(dataShape) || !validGPUShape(indicesShape) {
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
	if !validGPUShape(outShape) {
		return false, nil
	}

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

func (m *metalEvaluation) tryConv(e *evaluation, step operators.Step) (handled bool, err error) {
	var (
		xIdx    = step.Inputs[0]
		wIdx    = step.Inputs[1]
		hasBias = len(step.Inputs) > 2 && step.Inputs[2] >= 0
		xType   DataType
		wType   DataType
		xShape  []int64
		wShape  []int64
		bShape  []int64
	)

	xType, xShape = m.valueMeta(e, xIdx)
	wType, wShape = m.valueMeta(e, wIdx)
	if xType != Float32 || wType != Float32 || m.isReducedPrecision(xIdx) {
		return false, nil
	}

	if !validGPUShape(xShape) || !validGPUShape(wShape) {
		return false, nil
	}

	bIdx := -1
	if hasBias {
		bIdx = step.Inputs[2]
		var bType DataType
		bType, bShape = m.valueMeta(e, bIdx)
		if bType != Float32 || !validGPUShape(bShape) {
			return false, nil
		}
	}

	shapes := [][]int64{xShape, wShape}
	if hasBias {
		shapes = append(shapes, bShape)
	}

	var (
		outShape []int64
		convCfg  tensor.Conv2DConfig
	)

	if outShape, convCfg, _, err = step.CNN.Shape("Conv", shapes); err != nil || !validGPUShape(outShape) {
		return false, nil
	}

	gpuCfg := gpu.Conv2DConfig{
		Strides:   convCfg.Strides,
		Pads:      convCfg.Pads,
		Dilations: convCfg.Dilations,
		Groups:    convCfg.Groups,
	}

	if _, err = gpu.Conv2DShape(xShape, wShape, gpuCfg); err != nil {
		return false, nil
	}

	if !conv2DGeometryFitsInt32(xShape, wShape, outShape, gpuCfg) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		input   *gpu.Tensor
		weights *gpu.Tensor
		bias    *gpu.Tensor
		dst     *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, xIdx); err != nil {
		return true, err
	}

	if weights, err = m.ensureDeviceInput(e, step.Index, 1, wIdx); err != nil {
		return true, err
	}

	if hasBias {
		if bias, err = m.ensureDeviceInput(e, step.Index, 2, bIdx); err != nil {
			return true, err
		}
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal Conv destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.Conv2D(input, weights, bias, dst, gpuCfg); err != nil {
		return true, fmt.Errorf("encode Metal Conv: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryBatchNorm(e *evaluation, step operators.Step) (handled bool, err error) {
	if len(step.Outputs) > 1 && step.Outputs[1] >= 0 {
		return false, nil
	}

	if len(step.Inputs) != 5 {
		return false, nil
	}

	epsilon := step.CNN.Epsilon
	if !(epsilon >= 0 && epsilon <= math.MaxFloat32) {
		return false, nil
	}

	shapes := make([][]int64, len(step.Inputs))
	for slot, idx := range step.Inputs {
		if idx < 0 {
			return false, nil
		}

		var (
			inType  DataType
			inShape []int64
		)

		inType, inShape = m.valueMeta(e, idx)
		if inType != Float32 || m.isReducedPrecision(idx) || !validGPUShape(inShape) {
			return false, nil
		}

		shapes[slot] = inShape
	}

	var outShape []int64
	if outShape, _, _, err = step.CNN.Shape("BatchNormalization", shapes); err != nil || !validGPUShape(outShape) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		gpuInputs [5]*gpu.Tensor
		dst       *gpu.Tensor
	)

	for slot, idx := range step.Inputs {
		if gpuInputs[slot], err = m.ensureDeviceInput(e, step.Index, slot, idx); err != nil {
			return true, err
		}
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal BatchNormalization destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.BatchNorm(
		gpuInputs[0],
		gpuInputs[1],
		gpuInputs[2],
		gpuInputs[3],
		gpuInputs[4],
		dst,
		epsilon,
	); err != nil {
		return true, fmt.Errorf("encode Metal BatchNormalization: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryPool2D(e *evaluation, step operators.Step) (handled bool, err error) {
	if step.Operator == "MaxPool" && len(step.Outputs) > 1 && step.Outputs[1] >= 0 {
		return false, nil
	}

	var (
		xIdx   = step.Inputs[0]
		xType  DataType
		xShape []int64
	)

	xType, xShape = m.valueMeta(e, xIdx)
	if xType != Float32 || m.isReducedPrecision(xIdx) || len(xShape) != 4 || !validGPUShape(xShape) {
		return false, nil
	}

	var (
		outShape []int64
		poolCfg  tensor.Pool2DConfig
	)

	if outShape, _, poolCfg, err = step.CNN.Shape(step.Operator, [][]int64{xShape}); err != nil || !validGPUShape(outShape) {
		return false, nil
	}

	var gpuCfg gpu.Pool2DConfig
	if step.Operator == "GlobalAveragePool" {
		gpuCfg = gpu.Pool2DConfig{
			Kernel:    [2]int64{xShape[2], xShape[3]},
			Strides:   [2]int64{1, 1},
			Dilations: [2]int64{1, 1},
		}
	} else {
		gpuCfg = gpu.Pool2DConfig{
			Kernel:          poolCfg.Kernel,
			Strides:         poolCfg.Strides,
			Pads:            poolCfg.Pads,
			Dilations:       poolCfg.Dilations,
			CeilMode:        poolCfg.CeilMode,
			CountIncludePad: poolCfg.CountIncludePad,
		}
	}

	if _, err = gpu.Pool2DShape(xShape, gpuCfg); err != nil {
		return false, nil
	}

	if !pool2DGeometryFitsInt32(xShape, outShape, gpuCfg) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	var (
		input *gpu.Tensor
		dst   *gpu.Tensor
	)

	if input, err = m.ensureDeviceInput(e, step.Index, 0, xIdx); err != nil {
		return true, err
	}

	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal %s destination: %w", step.Operator, err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if step.Operator == "MaxPool" {
		err = m.scope.MaxPool2D(input, dst, gpuCfg)
	} else {
		err = m.scope.AvgPool2D(input, dst, gpuCfg)
	}

	if err != nil {
		return true, fmt.Errorf("encode Metal %s: %w", step.Operator, err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func (m *metalEvaluation) tryConcat(e *evaluation, step operators.Step) (handled bool, err error) {
	if len(step.Inputs) == 0 {
		return false, nil
	}

	var (
		rank     int
		axis     int
		outShape []int64
	)

	for slot, idx := range step.Inputs {
		if idx < 0 {
			return false, nil
		}

		var (
			inType  DataType
			inShape []int64
		)

		inType, inShape = m.valueMeta(e, idx)
		if inType != Float32 || m.isReducedPrecision(idx) || !validGPUShape(inShape) || len(inShape) > 6 {
			return false, nil
		}

		if slot == 0 {
			rank = len(inShape)
			if axis, err = tensor.NormalizeAxis(step.Axis, rank); err != nil {
				return false, nil
			}

			outShape = slices.Clone(inShape)
			continue
		}

		if len(inShape) != rank {
			return false, nil
		}

		for dim := range rank {
			if dim != axis && inShape[dim] != outShape[dim] {
				return false, nil
			}
		}

		if inShape[axis] > math.MaxInt64-outShape[axis] {
			return false, nil
		}

		outShape[axis] += inShape[axis]
	}

	if !validGPUShape(outShape) {
		return false, nil
	}

	if err = m.validateAndReserve(e, outShape); err != nil {
		return true, err
	}

	inputs := make([]*gpu.Tensor, len(step.Inputs))
	for slot, idx := range step.Inputs {
		if inputs[slot], err = m.ensureDeviceInput(e, step.Index, slot, idx); err != nil {
			return true, err
		}
	}

	var dst *gpu.Tensor
	if dst, err = m.cache.acquire(step.Index, 0, outShape); err != nil {
		return true, fmt.Errorf("allocate Metal Concat destination: %w", err)
	}

	if err = m.ensureScope(); err != nil {
		return true, err
	}

	if err = m.scope.Concat(inputs, dst, axis); err != nil {
		return true, fmt.Errorf("encode Metal Concat: %w", err)
	}

	return true, m.publishGPUValue(e, step.Outputs[0], dst, outShape)
}

func conv2DGeometryFitsInt32(xShape, wShape, outShape []int64, cfg gpu.Conv2DConfig) (ok bool) {
	return spatialGeometryFitsInt32(
		xShape[2], xShape[3],
		outShape[2], outShape[3],
		wShape[2], wShape[3],
		cfg.Strides, cfg.Pads, cfg.Dilations,
	)
}

func pool2DGeometryFitsInt32(xShape, outShape []int64, cfg gpu.Pool2DConfig) (ok bool) {
	return spatialGeometryFitsInt32(
		xShape[2], xShape[3],
		outShape[2], outShape[3],
		cfg.Kernel[0], cfg.Kernel[1],
		cfg.Strides, cfg.Pads, cfg.Dilations,
	)
}

func spatialGeometryFitsInt32(
	inH, inW, outH, outW, kernelH, kernelW int64,
	strides [2]int64,
	pads [4]int64,
	dilations [2]int64,
) (ok bool) {
	return inH <= math.MaxInt32 && inW <= math.MaxInt32 &&
		outH <= math.MaxInt32 && outW <= math.MaxInt32 &&
		kernelH <= math.MaxInt32 && kernelW <= math.MaxInt32 &&
		strides[0] <= math.MaxInt32 && strides[1] <= math.MaxInt32 &&
		pads[0] <= math.MaxInt32 && pads[1] <= math.MaxInt32 &&
		dilations[0] <= math.MaxInt32 && dilations[1] <= math.MaxInt32 &&
		inH+pads[0] <= math.MaxInt32 && inW+pads[1] <= math.MaxInt32 &&
		pads[0]+dilations[0] <= math.MaxInt32 && pads[1]+dilations[1] <= math.MaxInt32 &&
		(outH-1)*strides[0]+(kernelH-1)*dilations[0] <= math.MaxInt32 &&
		(outW-1)*strides[1]+(kernelW-1)*dilations[1] <= math.MaxInt32
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
	uploadShape := shape
	if len(uploadShape) == 0 {
		uploadShape = []int64{1}
	}

	if out, err = m.cache.acquire(stepIndex, 10+slot, uploadShape); err != nil {
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
		err = errors.Join(err, m.scope.Close())
		m.scope = nil
	}

	for _, view := range m.views {
		if view != nil {
			err = errors.Join(err, view.Close())
		}
	}

	m.views = nil
	return err
}
