//! Seeded Perl 5.32's string hash and the key order of a small hash, for output that VEP takes
//! from `keys %hash`. The reference runs use `PERL_HASH_SEED=0 PERL_PERTURB_KEYS=0`; Perl 5.32 on
//! 64-bit hashes keys of up to 24 bytes with SBOX32 and longer ones with STADTX
//! (`hv_func.h`, `sbox32_hash.h`, `stadtx_hash.h`), and with perturbation off a hash of up to
//! 8 keys lists its 8 buckets from the last to the first, the newest key first within one.
//! (An unseeded Perl, as in AIM's Docker image, orders such keys differently on every run.)

use std::sync::OnceLock;

const SBOX32_MAX_LEN: usize = 24;

/// The SBOX32 state for the all-zero seed (`sbox32_seed_state96`).
fn sbox32_state() -> &'static [u32] {
    static STATE: OnceLock<Vec<u32>> = OnceLock::new();
    STATE.get_or_init(|| {
        let (mut s0, mut s1, mut s2) = (0x6873_6168u32, 0x786f_6273u32, 0x646f_6f67u32);
        for _ in 0..5 {
            // SBOX32_MIX3
            s0 = s0.rotate_left(16).wrapping_sub(s2);
            s1 = s1.rotate_right(13) ^ s2;
            s2 = s2.rotate_left(17).wrapping_add(s1);
            s0 = s0.rotate_right(2).wrapping_add(s1);
            s1 = s1.rotate_right(17).wrapping_sub(s0);
            s2 = s2.rotate_right(7) ^ s0;
        }
        let mut xorshift = || {
            let t = s0 ^ (s0 << 10);
            s0 = s1;
            s1 = s2;
            s2 = (s2 ^ (s2 >> 26)) ^ (t ^ (t >> 5));
            s2
        };
        let mut state = vec![0u32; 1 + 256 * SBOX32_MAX_LEN];
        for x in state.iter_mut().skip(1) {
            *x = xorshift();
        }
        state[0] = xorshift();
        state
    })
}

/// The STADTX state for the all-zero seed (`stadtx_seed_state`).
fn stadtx_state() -> [u64; 4] {
    fn scramble(mut v: u64, prime: u64) -> u64 {
        v ^= v >> 13;
        v ^= v << 35;
        v ^= v >> 30;
        v = v.wrapping_mul(prime);
        v ^= v >> 19;
        v ^= v << 15;
        v ^= v >> 46;
        v
    }
    let mut s = [
        0x43f6_a888_5a30_8d31u64,
        0x3198_a2e0_3707_344a,
        0x4093_8222_99f3_1d00,
        0x82ef_a98e_c4e6_c894,
    ];
    s[0] = scramble(scramble(s[0], 0x8011_7884_6e89_9d17), 0xdd51_e5d1_c9a5_a151);
    s[1] = scramble(scramble(s[1], 0x93a7_d6c8_c62e_4835), 0x8033_40f3_6895_c2b5);
    s[2] = scramble(scramble(s[2], 0xbea9_344e_b756_5eeb), 0xcd95_d1e5_09b9_95cd);
    s[3] = scramble(scramble(s[3], 0x9999_7919_77e3_0c13), 0xaab8_b6b0_5abf_c6cd);
    s
}

fn le64(k: &[u8]) -> u64 {
    u64::from_le_bytes(k[..8].try_into().unwrap())
}
fn le32(k: &[u8]) -> u64 {
    u32::from_le_bytes(k[..4].try_into().unwrap()) as u64
}
fn le16(k: &[u8]) -> u64 {
    u16::from_le_bytes(k[..2].try_into().unwrap()) as u64
}

/// `stadtx_hash_with_state`.
fn stadtx(key: &[u8]) -> u64 {
    const K0: u64 = 0xb89b_0f8e_1655_514f;
    const K1: u64 = 0x8c6f_7360_11bd_5127;
    const K2: u64 = 0x8f29_bd94_edce_7b39;
    const K3: u64 = 0x9c1b_8e1e_9628_323f;
    const K2_32: u64 = 0x8029_10e3;
    const K3_32: u64 = 0x819b_13af;
    const K4_32: u64 = 0x91cb_27e5;
    const K5_32: u64 = 0xc1a2_69c1;
    let state = stadtx_state();
    let n = key.len() as u64;
    let mut v0 = state[0] ^ (n + 1).wrapping_mul(K0);
    let mut v1 = state[1] ^ (n + 2).wrapping_mul(K1);
    let mut k = key;
    if key.len() < 32 {
        for _ in 0..(k.len() >> 3) {
            v0 = v0.wrapping_add(le64(k).wrapping_mul(K3));
            v0 = v0.rotate_right(17) ^ v1;
            v1 = v1.rotate_right(53).wrapping_add(v0);
            k = &k[8..];
        }
        let r = key.len() & 7;
        if r == 0 {
            v1 = v1.rotate_left(32) ^ 0xFF;
        } else if r >= 4 {
            if r == 7 {
                v0 = v0.wrapping_add((k[6] as u64) << 32);
            }
            if r >= 6 {
                v1 = v1.wrapping_add((k[5] as u64) << 48);
            }
            if r >= 5 {
                v0 = v0.wrapping_add((k[4] as u64) << 16);
            }
            v1 = v1.wrapping_add(le32(k));
        } else if r >= 2 {
            if r == 3 {
                v0 = v0.wrapping_add((k[2] as u64) << 48);
            }
            v1 = v1.wrapping_add(le16(k));
        } else {
            v0 = v0.wrapping_add(k[0] as u64);
            v1 = v1.rotate_left(32) ^ 0xFF;
        }
        v1 ^= v0;
        v0 = v0.rotate_right(33).wrapping_add(v1);
        v1 = v1.rotate_left(17) ^ v0;
        v0 = v0.rotate_left(43).wrapping_add(v1);
        v1 = v1.rotate_left(31).wrapping_sub(v0);
        v0 = v0.rotate_left(13) ^ v1;
        v1 = v1.wrapping_sub(v0);
        v0 = v0.rotate_left(41).wrapping_add(v1);
        v1 = v1.rotate_left(37) ^ v0;
        v0 = v0.rotate_right(39).wrapping_add(v1);
        v1 = v1.rotate_right(15).wrapping_add(v0);
        v0 = v0.rotate_left(15) ^ v1;
        v1 = v1.rotate_right(5);
        return v0 ^ v1;
    }
    let mut v2 = state[2] ^ (n + 3).wrapping_mul(K2);
    let mut v3 = state[3] ^ (n + 4).wrapping_mul(K3);
    while k.len() >= 32 {
        v0 = v0.wrapping_add(le64(k).wrapping_mul(K2_32));
        v0 = v0.rotate_left(57) ^ v3;
        v1 = v1.wrapping_add(le64(&k[8..]).wrapping_mul(K3_32));
        v1 = v1.rotate_left(63) ^ v2;
        v2 = v2.wrapping_add(le64(&k[16..]).wrapping_mul(K4_32));
        v2 = v2.rotate_right(47).wrapping_add(v0);
        v3 = v3.wrapping_add(le64(&k[24..]).wrapping_mul(K5_32));
        v3 = v3.rotate_right(11).wrapping_sub(v1);
        k = &k[32..];
    }
    // the remaining length (under 32), not reduced by the words below
    let rem = k.len() as u64;
    let words = k.len() >> 3;
    if words >= 3 {
        v0 = v0.wrapping_add(le64(k).wrapping_mul(K2_32));
        k = &k[8..];
        v0 = v0.rotate_left(57) ^ v3;
    }
    if words >= 2 {
        v1 = v1.wrapping_add(le64(k).wrapping_mul(K3_32));
        k = &k[8..];
        v1 = v1.rotate_left(63) ^ v2;
    }
    if words >= 1 {
        v2 = v2.wrapping_add(le64(k).wrapping_mul(K4_32));
        k = &k[8..];
        v2 = v2.rotate_right(47).wrapping_add(v0);
    }
    v3 = v3.rotate_right(11).wrapping_sub(v1);
    v0 ^= (rem + 1).wrapping_mul(K3);
    match k.len() & 7 {
        7 | 6 => {
            if k.len() == 7 {
                v1 = v1.wrapping_add(k[6] as u64);
            }
            v2 = v2.wrapping_add(le16(&k[4..]));
            v3 = v3.wrapping_add(le32(k));
        }
        5 | 4 => {
            if k.len() == 5 {
                v1 = v1.wrapping_add(k[4] as u64);
            }
            v2 = v2.wrapping_add(le32(k));
        }
        3 | 2 => {
            if k.len() == 3 {
                v3 = v3.wrapping_add(k[2] as u64);
            }
            v1 = v1.wrapping_add(le16(k));
        }
        1 => {
            v2 = v2.wrapping_add(k[0] as u64);
            v3 = v3.rotate_left(32) ^ 0xFF;
        }
        _ => v3 = v3.rotate_left(32) ^ 0xFF,
    }
    v1 = v1.wrapping_sub(v2);
    v0 = v0.rotate_right(19);
    v1 = v1.wrapping_sub(v0);
    v1 = v1.rotate_right(53);
    v3 ^= v1;
    v0 = v0.wrapping_sub(v3);
    v3 = v3.rotate_left(43);
    v0 = v0.wrapping_add(v3);
    v0 = v0.rotate_right(3);
    v3 = v3.wrapping_sub(v0);
    v2 = v2.rotate_right(43).wrapping_sub(v3);
    v2 = v2.rotate_left(55) ^ v0;
    v1 = v1.wrapping_sub(v2);
    v3 = v3.rotate_right(7).wrapping_sub(v2);
    v2 = v2.rotate_right(31);
    v3 = v3.wrapping_add(v2);
    v2 = v2.wrapping_sub(v1);
    v3 = v3.rotate_right(39);
    v2 ^= v3;
    v3 = v3.rotate_right(17) ^ v2;
    v1 = v1.wrapping_add(v3);
    v1 = v1.rotate_right(9);
    v2 ^= v1;
    v2 = v2.rotate_left(24);
    v3 ^= v2;
    v3 = v3.rotate_right(59);
    v0 = v0.rotate_right(1).wrapping_sub(v1);
    v0 ^ v1 ^ v2 ^ v3
}

/// Seeded Perl 5.32's `PERL_HASH` of a key (`Hash::Util::hash_value`).
pub fn hash(key: &[u8]) -> u32 {
    if key.len() <= SBOX32_MAX_LEN {
        let s = sbox32_state();
        let mut h = s[0];
        for (i, &b) in key.iter().enumerate() {
            h ^= s[1 + 256 * i + b as usize];
        }
        h
    } else {
        stadtx(key) as u32
    }
}

/// The order `keys` lists a hash built by inserting `keys` in order (repeats ignored), for at
/// most 8 distinct keys (one 8-bucket array).
pub fn keys_order<'a>(keys: &[&'a str]) -> Vec<&'a str> {
    let mut distinct: Vec<&str> = Vec::new();
    for k in keys {
        if !distinct.contains(k) {
            distinct.push(k);
        }
    }
    assert!(distinct.len() <= 8, "keys_order: more than 8 keys");
    // by bucket from the last, the newest first within one
    let mut order: Vec<(u32, usize, &str)> = distinct
        .iter()
        .enumerate()
        .map(|(i, k)| (hash(k.as_bytes()) & 7, i, *k))
        .collect();
    order.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    order.into_iter().map(|(_, _, k)| k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_hash_values() {
        // `PERL_HASH_SEED=0 perl -MHash::Util=hash_value -e 'print hash_value($k)'`
        assert_eq!(hash(b"A"), 1820471646);
        assert_eq!(hash(b"C"), 937701381);
        assert_eq!(hash(b"G"), 2581985614);
        assert_eq!(hash(b"T"), 923619741);
        assert_eq!(hash(b"-"), 58948784);
        assert_eq!(hash(b"AT"), 2741784179);
        assert_eq!(hash(b"CA"), 1941917075);
        // STADTX, short and long paths (checked on 3,000 random keys of 1-80 bytes)
        assert_eq!(hash(b"Ga11AN1/G-ATGTCTA-1aA/a1NN/GCN"), 1255996747);
        assert_eq!(
            hash(b"T/CG//C-CAaCGNNC11-AGa-AGGTACTC-AACCaG/1T/-a1GNNTCa//N/aN"),
            523043881
        );
    }

    #[test]
    fn seeded_key_order() {
        // `my %h = map {$_ => 1} @k; keys %h`
        assert_eq!(keys_order(&["T", "G"]), vec!["G", "T"]);
        assert_eq!(keys_order(&["A", "G"]), vec!["G", "A"]);
        assert_eq!(keys_order(&["G", "A"]), vec!["A", "G"]);
        assert_eq!(keys_order(&["-", "AT"]), vec!["AT", "-"]);
        assert_eq!(keys_order(&["A", "C"]), vec!["A", "C"]);
    }
}
