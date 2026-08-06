//! Deterministic pseudo-random numbers, without a dependency.

/// SplitMix64 — a small deterministic PRNG.
///
/// Used wherever a kernel needs reproducible randomness without pulling in a
/// dependency: the Louvain node visit order, and the rewiring edge picks.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, bound)` via Lemire's debiased multiply-shift.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let mut product = (self.next_u64() as u128).wrapping_mul(bound as u128);
        let mut low = product as u64;
        if low < bound {
            let threshold = (u64::MAX - bound + 1) % bound;
            while low < threshold {
                product = (self.next_u64() as u128).wrapping_mul(bound as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    pub(crate) fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = self.below((i + 1) as u64) as usize;
            items.swap(i, j);
        }
    }
}
