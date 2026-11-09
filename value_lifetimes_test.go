package onnx

import (
	"context"
	"slices"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func reuseWire(t *testing.T) (out *onnxpb.ModelProto) {
	t.Helper()
	var wire onnxpb.ModelProto
	proto.Merge(&wire, modelWire(t, "profile/identity"))
	wire.Graph.Node = []*onnxpb.NodeProto{
		{OpType: proto.String("Add"), Input: []string{"x", "x"}, Output: []string{"a"}},
		{OpType: proto.String("Identity"), Input: []string{"a"}, Output: []string{"alias"}},
		{OpType: proto.String("Add"), Input: []string{"x", "x"}, Output: []string{"b"}},
		{OpType: proto.String("Add"), Input: []string{"alias", "b"}, Output: []string{"join"}},
		{OpType: proto.String("Add"), Input: []string{"join", "x"}, Output: []string{"c"}},
		{OpType: proto.String("Add"), Input: []string{"c", "x"}, Output: []string{"y"}},
	}
	return &wire
}

func TestValueLifetimesConsumerOccurrencesAndAliasRetirement(t *testing.T) {
	plan := loadedModel(t, modelBytes(t, reuseWire(t)), Options{}).plan
	lifetimes := plan.lifetimes
	input, left, alias := plan.indices["x"], plan.indices["a"], plan.indices["alias"]
	if lifetimes.consumers[input] != 6 || lifetimes.consumers[left] != 1 || lifetimes.consumers[alias] != 1 {
		t.Fatalf("consumer occurrences: %v", lifetimes.consumers)
	}

	if lifetimes.owners[alias] != left || !slices.Equal(lifetimes.members[left], []int{left, alias}) {
		t.Fatalf("Identity storage group: %+v", lifetimes)
	}

	if !lifetimes.pinned[input] || !lifetimes.pinned[plan.indices["y"]] || len(lifetimes.retire[1]) != 0 {
		t.Fatal("source/output pinning or early alias retirement")
	}

	if !slices.Equal(lifetimes.retire[3], []int{left, plan.indices["b"]}) {
		t.Fatalf("join retirement: %v", lifetimes.retire[3])
	}

	var members, retired int
	for _, group := range lifetimes.members {
		if cap(group) != len(group) {
			t.Fatal("membership storage has unaccounted append capacity")
		}

		members += len(group)
	}
	for _, group := range lifetimes.retire {
		if cap(group) != len(group) {
			t.Fatal("retirement storage has unaccounted append capacity")
		}

		retired += len(group)
	}
	if members != len(plan.values) || retired > members {
		t.Fatal("lifetime payload exceeded its reserved cardinality")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := newValueLifetimes(cancelled, plan); err != context.Canceled {
		t.Fatalf("cancelled lifetime compilation: %v", err)
	}
}

func TestValueLifetimesPinsEveryDeclaredAliasOutput(t *testing.T) {
	wire := reuseWire(t)
	alias := proto.Clone(wire.Graph.Output[0]).(*onnxpb.ValueInfoProto)
	alias.Name = proto.String("alias")
	wire.Graph.Output = append(wire.Graph.Output, alias)
	plan := loadedModel(t, modelBytes(t, wire), Options{}).plan
	owner := plan.indices["a"]
	if !plan.lifetimes.pinned[owner] {
		t.Fatal("declared output did not pin its backing owner")
	}

	for _, retired := range plan.lifetimes.retire {
		if slices.Contains(retired, owner) {
			t.Fatal("declared alias output can be recycled")
		}
	}
}
