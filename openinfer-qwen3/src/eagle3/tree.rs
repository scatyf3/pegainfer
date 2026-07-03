//! Static draft-tree topology for EAGLE-3 tree speculative decoding.
//!
//! v1 is a *static* tree: the shape (per-depth branching) is fixed at build time, so
//! the parent/depth arrays and the packed ancestor mask are constants built once —
//! CUDA-graph friendly, unlike a probability-driven dynamic tree.
//!
//! Node 0 is the ROOT = the round's current confirmed token; every other node is a
//! drafted candidate. Node `i` may attend exactly its ancestor chain
//! `[i, parent[i], …, 0]` (plus the whole committed prefix). Verify forwards all
//! nodes at once under this mask; accept walks the longest root→leaf path whose
//! tokens all match the target argmax.

/// Default v1 draft tree: depth 3, branching 2 at every level → `1+2+4+8 = 15`
/// nodes (14 drafts, verify span 15). A conservative, uniform starting shape;
/// the per-depth branching is the main tuning knob (accept decays with depth, so
/// widening shallow levels / narrowing deep ones is the expected next step).
pub(crate) const EAGLE3_DEFAULT_TREE_BRANCHING: &[usize] = &[2, 2, 2];

/// A static draft tree in BFS node order (root = 0, then depth 1 left→right, …),
/// which guarantees `parent[i] < i` for every non-root node — the topological order
/// the per-depth draft rollout and the packed mask both rely on.
pub(crate) struct Eagle3Tree {
    /// Per-node parent index. `parent[0] == 0` is the root's self-parent sentinel.
    parent: Vec<usize>,
    /// Per-node depth. `depth[0] == 0`.
    depth: Vec<usize>,
    /// `levels[d]` = node indices at depth `d`; `levels[d]` is the frontier the
    /// draft rollout forwards together at step `d`.
    levels: Vec<Vec<usize>>,
}

impl Eagle3Tree {
    /// Build from a per-depth branching factor: `branching[d]` children per depth-`d`
    /// node. Total nodes = `1 + b₀ + b₀b₁ + b₀b₁b₂ + …`.
    pub(crate) fn from_branching(branching: &[usize]) -> Self {
        let mut parent = vec![0usize];
        let mut depth = vec![0usize];
        let mut levels = vec![vec![0usize]];
        for (d, &b) in branching.iter().enumerate() {
            let mut level = Vec::new();
            for p in levels[d].clone() {
                for _ in 0..b {
                    let id = parent.len();
                    parent.push(p);
                    depth.push(d + 1);
                    level.push(id);
                }
            }
            levels.push(level);
        }
        Self {
            parent,
            depth,
            levels,
        }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        self.parent.len()
    }

    pub(crate) fn max_depth(&self) -> usize {
        self.levels.len() - 1
    }

    pub(crate) fn node_depth(&self, node: usize) -> usize {
        self.depth[node]
    }

    pub(crate) fn parent_of(&self, node: usize) -> usize {
        self.parent[node]
    }

    /// Node indices at depth `d` (the frontier forwarded together in draft step `d`).
    pub(crate) fn level(&self, d: usize) -> &[usize] {
        &self.levels[d]
    }

    /// Ancestor chain of `node` from itself up to and including the root:
    /// `[node, parent, …, 0]`. Exactly the set `node` may attend within the tree.
    pub(crate) fn ancestors(&self, node: usize) -> Vec<usize> {
        let mut chain = vec![node];
        let mut cur = node;
        while cur != 0 {
            cur = self.parent[cur];
            chain.push(cur);
        }
        chain
    }

    /// Bit-packed `[num_nodes, num_nodes]` intra-tree attention mask in the layout
    /// `single_prefill_nhd_custom_mask_into` expects: bit `i * num_nodes + j` set
    /// iff `j` is an ancestor-or-self of `i` (i.e. query node `i` may attend key
    /// node `j`), LSB-first within each byte. This is the tree block only; a full
    /// verify mask over `[prefix | tree]` prepends an all-ones prefix region so
    /// every node also sees the committed context.
    pub(crate) fn packed_ancestor_mask(&self) -> Vec<u8> {
        let n = self.num_nodes();
        let mut mask = vec![0u8; (n * n).div_ceil(8)];
        for i in 0..n {
            for j in self.ancestors(i) {
                let bit = i * n + j;
                mask[bit / 8] |= 1u8 << (bit % 8);
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attends(mask: &[u8], n: usize, i: usize, j: usize) -> bool {
        let bit = i * n + j;
        (mask[bit / 8] >> (bit % 8)) & 1 == 1
    }

    #[test]
    fn branching_shape_and_counts() {
        let t = Eagle3Tree::from_branching(&[2, 2, 2]);
        assert_eq!(t.num_nodes(), 15); // 1 + 2 + 4 + 8
        assert_eq!(t.max_depth(), 3);
        assert_eq!(t.level(0), &[0]);
        assert_eq!(t.level(1).len(), 2);
        assert_eq!(t.level(2).len(), 4);
        assert_eq!(t.level(3).len(), 8);
    }

    #[test]
    fn parent_is_topological() {
        let t = Eagle3Tree::from_branching(&[3, 2]);
        assert_eq!(t.num_nodes(), 1 + 3 + 6);
        for node in 1..t.num_nodes() {
            assert!(t.parent_of(node) < node, "parent[{node}] must precede it");
            assert_eq!(t.node_depth(node), t.node_depth(t.parent_of(node)) + 1);
        }
    }

    #[test]
    fn ancestors_reach_root() {
        let t = Eagle3Tree::from_branching(&[2, 2, 2]);
        // A deepest node's chain is [leaf, d2, d1, root] — length depth+1, ending at 0.
        let leaf = *t.level(3).last().unwrap();
        let chain = t.ancestors(leaf);
        assert_eq!(chain.len(), t.node_depth(leaf) + 1);
        assert_eq!(*chain.last().unwrap(), 0);
        assert_eq!(chain[0], leaf);
    }

    #[test]
    fn mask_encodes_ancestors_only() {
        // Smallest branching tree: root + two children.
        let t = Eagle3Tree::from_branching(&[2]);
        let n = t.num_nodes();
        assert_eq!(n, 3);
        let mask = t.packed_ancestor_mask();

        // Root (0) attends only itself.
        assert!(attends(&mask, n, 0, 0));
        assert!(!attends(&mask, n, 0, 1));
        assert!(!attends(&mask, n, 0, 2));

        // Child 1 attends itself + root, NOT its sibling 2.
        assert!(attends(&mask, n, 1, 1));
        assert!(attends(&mask, n, 1, 0));
        assert!(!attends(&mask, n, 1, 2));

        // Child 2 attends itself + root, NOT sibling 1.
        assert!(attends(&mask, n, 2, 2));
        assert!(attends(&mask, n, 2, 0));
        assert!(!attends(&mask, n, 2, 1));
    }

    #[test]
    fn mask_row_popcount_equals_depth_plus_one() {
        let t = Eagle3Tree::from_branching(&[2, 2, 2]);
        let n = t.num_nodes();
        let mask = t.packed_ancestor_mask();
        // Each node attends exactly its ancestor chain: depth+1 keys.
        for i in 0..n {
            let attended = (0..n).filter(|&j| attends(&mask, n, i, j)).count();
            assert_eq!(attended, t.node_depth(i) + 1, "node {i} attend count");
        }
    }
}
