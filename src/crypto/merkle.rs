use crate::crypto::hash::sha3_hash;

pub struct MerkleTree {
    pub root: Vec<u8>,
    nodes: Vec<Vec<u8>>,
}

impl MerkleTree {
    pub fn new(leaves: Vec<Vec<u8>>) -> Self {
        if leaves.is_empty() {
            return Self {
                root: vec![],
                nodes: vec![],
            };
        }
        
        let mut nodes = leaves;
        
        while nodes.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in nodes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    [&chunk[0][..], &chunk[1][..]].concat()
                } else {
                    chunk[0].clone()
                };
                next_level.push(sha3_hash(&combined));
            }
            
            nodes = next_level;
        }
        
        Self {
            root: nodes[0].clone(),
            nodes,
        }
    }
}