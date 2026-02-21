package main

import (
	"math"
	"testing"
)

// helper: build a field + tokenizer from test docs
func makeTestFieldAndTok() (*CooccurField, *EvolvingTokenizer, []string) {
	docs := []string{
		"Hello world hello world",
		"This is a test sentence",
		"Hello hello hello world test",
	}
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)
	return field, tok, docs
}

// ============================================================
// 4-GRAM TESTS
// ============================================================

func TestBuildPopulatesFourgram(t *testing.T) {
	field, _, _ := makeTestFieldAndTok()
	if len(field.FourgramByCtx) == 0 {
		t.Error("BuildFromCorpus should populate 4-gram map")
	}
}

func TestFourgramCountsMatchExpected(t *testing.T) {
	// Deterministic test: single doc with known 4-gram
	docs := []string{"abcde"}
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)

	// Encode and check 4-gram: tokens[0],tokens[1],tokens[2] → tokens[3]
	ids := tok.Encode("abcde")
	if len(ids) < 5 { // BOS + a + b + c + d + e + EOS
		t.Skipf("Tokenizer produced too few tokens: %d", len(ids))
	}
	// Skip BOS (ids[0]). Look for 4-gram [ids[1], ids[2], ids[3]] → ids[4]
	ctx := [3]int{ids[1], ids[2], ids[3]}
	if dist, ok := field.FourgramByCtx[ctx]; ok {
		if dist[ids[4]] < 1 {
			t.Error("4-gram count should be >= 1")
		}
	} else {
		t.Error("Expected 4-gram context not found")
	}
}

func TestIngestTokensAddsFourgram(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	before := len(field.FourgramByCtx)
	// Ingest a new unique sequence
	newIDs := tok.Encode("zyxwvut")
	field.IngestTokens(newIDs)
	after := len(field.FourgramByCtx)
	if after <= before {
		t.Errorf("IngestTokens should add 4-gram entries: before=%d, after=%d", before, after)
	}
}

// ============================================================
// CO-OCCURRENCE WINDOW TESTS
// ============================================================

func TestBuildPopulatesCooccurWindow(t *testing.T) {
	field, _, _ := makeTestFieldAndTok()
	if len(field.CooccurWindow) == 0 {
		t.Error("BuildFromCorpus should populate co-occurrence window map")
	}
}

func TestCooccurWindowSymmetry(t *testing.T) {
	// If A and B appear within window, both A→B and B→A should exist
	docs := []string{"ab"}
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)

	ids := tok.Encode("ab")
	if len(ids) < 3 {
		t.Skipf("Too few tokens: %d", len(ids))
	}
	// ids[1] and ids[2] should co-occur (a and b)
	a, b := ids[1], ids[2]
	aNeighbors := field.CooccurWindow[a]
	bNeighbors := field.CooccurWindow[b]
	if aNeighbors == nil || aNeighbors[b] == 0 {
		t.Error("a should have b as co-occurrence neighbor")
	}
	if bNeighbors == nil || bNeighbors[a] == 0 {
		t.Error("b should have a as co-occurrence neighbor")
	}
}

func TestCooccurWindowRespectSize(t *testing.T) {
	// With window=2, tokens far apart should NOT co-occur
	oldWindow := CFG.CooccurWindowSize
	CFG.CooccurWindowSize = 1 // very small window
	defer func() { CFG.CooccurWindowSize = oldWindow }()

	docs := []string{"abcdefghij"} // 10 chars
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)

	ids := tok.Encode("abcdefghij")
	if len(ids) < 6 {
		t.Skipf("Too few tokens: %d", len(ids))
	}
	// With window=1, ids[1] (a) and ids[5] (e) should NOT co-occur
	a := ids[1]
	e := ids[5]
	if field.CooccurWindow[a] != nil && field.CooccurWindow[a][e] > 0 {
		t.Error("With window=1, distant tokens should not co-occur")
	}
}

func TestIngestTokensAddsCooccur(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	// Track a specific token's co-occurrence count
	totalBefore := 0.0
	for _, neighbors := range field.CooccurWindow {
		for _, cnt := range neighbors {
			totalBefore += cnt
		}
	}
	field.IngestTokens(tok.Encode("new text here"))
	totalAfter := 0.0
	for _, neighbors := range field.CooccurWindow {
		for _, cnt := range neighbors {
			totalAfter += cnt
		}
	}
	if totalAfter <= totalBefore {
		t.Error("IngestTokens should increase co-occurrence counts")
	}
}

// ============================================================
// USER BOOST TESTS
// ============================================================

func TestAbsorbUserWordsAddsBoosts(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	userIDs := tok.Encode("Hello")
	field.AbsorbUserWords(userIDs)

	found := false
	for _, id := range userIDs {
		if boost, ok := field.UserBoost[id]; ok && boost > 0 {
			found = true
		}
	}
	if !found {
		t.Error("AbsorbUserWords should set positive boosts for user tokens")
	}
}

func TestAbsorbUserWordsDecaysOld(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	// Set initial boost
	firstIDs := tok.Encode("Hello")
	field.AbsorbUserWords(firstIDs)
	oldBoost := field.UserBoost[firstIDs[1]] // skip BOS

	// Absorb different words — old boosts should decay
	secondIDs := tok.Encode("world")
	field.AbsorbUserWords(secondIDs)
	newBoost := field.UserBoost[firstIDs[1]]

	if newBoost >= oldBoost {
		t.Errorf("Old boosts should decay: old=%f, new=%f", oldBoost, newBoost)
	}
}

func TestDecayUserBoostReduces(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	userIDs := tok.Encode("test")
	field.AbsorbUserWords(userIDs)

	// Get max boost
	maxBefore := 0.0
	for _, v := range field.UserBoost {
		if v > maxBefore {
			maxBefore = v
		}
	}

	field.DecayUserBoost()

	maxAfter := 0.0
	for _, v := range field.UserBoost {
		if v > maxAfter {
			maxAfter = v
		}
	}

	if maxAfter >= maxBefore {
		t.Errorf("DecayUserBoost should reduce boosts: before=%f, after=%f", maxBefore, maxAfter)
	}
}

func TestDecayUserBoostRemovesSmall(t *testing.T) {
	field := NewCooccurField()
	field.UserBoost[42] = 0.005 // below threshold
	field.DecayUserBoost()

	if _, ok := field.UserBoost[42]; ok {
		t.Error("DecayUserBoost should remove boosts below 0.01")
	}
}

// ============================================================
// WEIGHTED INGESTION TESTS
// ============================================================

func TestIngestTokensWeightedHighWeight(t *testing.T) {
	docs := []string{"ab"}
	tok := NewEvolvingTokenizer(docs)

	field1 := NewCooccurField()
	field1.BuildFromCorpus(tok, docs)
	ids := tok.Encode("cd")
	field1.IngestTokensWeighted(ids, 1.0)

	field2 := NewCooccurField()
	field2.BuildFromCorpus(tok, docs)
	field2.IngestTokensWeighted(ids, 5.0)

	// Field2 should have higher counts for the ingested tokens
	if len(ids) >= 2 {
		c1 := field1.BigramByFirst[ids[0]][ids[1]]
		c2 := field2.BigramByFirst[ids[0]][ids[1]]
		if c2 <= c1 {
			t.Errorf("Weight 5.0 should produce higher counts than 1.0: c1=%f, c2=%f", c1, c2)
		}
	}
}

func TestIngestTokensWeightedZeroWeight(t *testing.T) {
	field, tok, docs := makeTestFieldAndTok()
	// Snapshot a bigram count
	ids := tok.Encode("Hello")
	var beforeCount float64
	if len(ids) >= 3 {
		beforeCount = field.BigramByFirst[ids[1]][ids[2]]
	}
	// Ingest with weight 0 — should not change
	field.IngestTokensWeighted(ids, 0.0)
	if len(ids) >= 3 {
		afterCount := field.BigramByFirst[ids[1]][ids[2]]
		if afterCount != beforeCount {
			t.Errorf("Weight 0.0 should not change counts: before=%f, after=%f", beforeCount, afterCount)
		}
	}
	_ = docs
}

// ============================================================
// SAMPLE_NEXT WITH NEW FEATURES
// ============================================================

func TestSampleNextUsesFourgram(t *testing.T) {
	// Build field from single doc so 4-gram is deterministic
	docs := []string{"abcde"}
	tok := NewEvolvingTokenizer(docs)
	field := NewCooccurField()
	field.BuildFromCorpus(tok, docs)

	ids := tok.Encode("abcde")
	if len(ids) < 6 {
		t.Skipf("Too few tokens: %d", len(ids))
	}
	// Context: last 3 tokens before 'e'
	ctx := ids[1:4] // a, b, c — should predict d
	expected := ids[4]
	// Sample many times; 4-gram should strongly prefer expected
	hits := 0
	for i := 0; i < 100; i++ {
		got := field.SampleNext(ctx, tok.VocabSize, 0.5) // low temp
		if got == expected {
			hits++
		}
	}
	if hits < 20 {
		t.Errorf("4-gram should predict next token more often, got %d/100 hits", hits)
	}
}

func TestSampleNextWithUserBoost(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	ids := tok.Encode("Hello")
	if len(ids) < 2 {
		t.Skip("Too few tokens")
	}

	// Boost a specific token heavily
	targetID := ids[1]
	field.UserBoost[targetID] = 10.0 // very strong boost

	// Sample with single-token context — boosted token should appear more
	ctx := ids[:1]
	hits := 0
	for i := 0; i < 200; i++ {
		got := field.SampleNext(ctx, tok.VocabSize, 1.0)
		if got == targetID {
			hits++
		}
	}
	if hits < 20 {
		t.Errorf("Strongly boosted token should appear frequently, got %d/200 hits", hits)
	}
}

// ============================================================
// INTEGRATION: BUILD → INGEST → SAMPLE PIPELINE
// ============================================================

func TestFullPipeline(t *testing.T) {
	field, tok, docs := makeTestFieldAndTok()

	// 1. Build from corpus — already done
	if !field.Built {
		t.Fatal("Field should be built")
	}

	// 2. Absorb user words
	userIDs := tok.Encode("Hello world")
	field.AbsorbUserWords(userIDs)
	if len(field.UserBoost) == 0 {
		t.Error("UserBoost should be non-empty after absorb")
	}

	// 3. Ingest new text weighted
	field.IngestTokensWeighted(tok.Encode("test words"), 1.5)

	// 4. Sample should produce valid token
	ctx := tok.Encode("Hello")
	if len(ctx) > 1 {
		nxt := field.SampleNext(ctx[1:], tok.VocabSize, 1.0)
		if nxt < 0 || nxt >= tok.VocabSize {
			t.Errorf("SampleNext returned invalid token: %d", nxt)
		}
	}

	// 5. Decay user boost — all values should shrink
	maxBefore := 0.0
	for _, v := range field.UserBoost {
		if v > maxBefore {
			maxBefore = v
		}
	}
	field.DecayUserBoost()
	maxAfter := 0.0
	for _, v := range field.UserBoost {
		if v > maxAfter {
			maxAfter = v
		}
	}
	if maxAfter >= maxBefore {
		t.Errorf("After decay, max boost should shrink: before=%f, after=%f", maxBefore, maxAfter)
	}

	// 6. Verify 4-gram and cooccur exist
	if len(field.FourgramByCtx) == 0 {
		t.Error("4-gram map should not be empty after build + ingest")
	}
	if len(field.CooccurWindow) == 0 {
		t.Error("CooccurWindow should not be empty after build + ingest")
	}

	_ = docs
}

// ============================================================
// CORPUS BLEND PARAMS
// ============================================================

func TestCooccurWindowSizeConfig(t *testing.T) {
	if CFG.CooccurWindowSize != 5 {
		t.Errorf("Default CooccurWindowSize should be 5, got %d", CFG.CooccurWindowSize)
	}
}

func TestUserBoostStrengthConfig(t *testing.T) {
	if math.Abs(CFG.UserBoostStrength-0.3) > 1e-6 {
		t.Errorf("Default UserBoostStrength should be 0.3, got %f", CFG.UserBoostStrength)
	}
}

func TestUserBoostDecayConfig(t *testing.T) {
	if math.Abs(CFG.UserBoostDecay-0.7) > 1e-6 {
		t.Errorf("Default UserBoostDecay should be 0.7, got %f", CFG.UserBoostDecay)
	}
}

// ============================================================
// REBUILD CLEARS NEW STRUCTURES
// ============================================================

func TestRebuildClearsFourgram(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	// Ingest extra to add unique 4-grams
	field.IngestTokens(tok.Encode("unique sequence of tokens here"))
	before := len(field.FourgramByCtx)

	// Rebuild with tiny docs — should clear
	field.BuildFromCorpus(tok, []string{"ab"})
	after := len(field.FourgramByCtx)
	if after >= before {
		t.Errorf("Rebuild should clear 4-gram map: before=%d, after=%d", before, after)
	}
}

func TestRebuildClearsCooccurWindow(t *testing.T) {
	field, tok, _ := makeTestFieldAndTok()
	before := len(field.CooccurWindow)

	field.BuildFromCorpus(tok, []string{"a"})
	after := len(field.CooccurWindow)
	if after >= before {
		t.Errorf("Rebuild should clear co-occur window: before=%d, after=%d", before, after)
	}
}
