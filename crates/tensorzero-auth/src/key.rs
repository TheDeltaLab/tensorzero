// Modified by Delta-AI under Apache 2.0
use std::sync::Arc;

use rand::distr::{Alphanumeric, SampleString};
use secrecy::{ExposeSecret, SecretString};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::postgres::KeyInfo;

/// A parsed TensorZero API key, with the long key hashed.
/// This does not contain the original long key
#[derive(Debug)]
pub struct TensorZeroApiKey {
    pub public_id: String,
    pub hashed_long_key: SecretString,
}

const SK_PREFIX: &str = "sk";
const T0_PREFIX: &str = "t0";
const SYNAPSE_KEY_PREFIX: &str = "sk-syn-v1-";
pub const PUBLIC_ID_LENGTH: usize = 12;
const LONG_KEY_LENGTH: usize = 48;

/// Securely generates a fresh API key
/// The `crate::postgres::create_key` function will use this to generate a fresh API key and insert it into the database.
pub(crate) fn secure_fresh_api_key() -> SecretString {
    // Use a cryptographically secure RNG, seeded from the OS
    let mut rng = rand::rng();
    let short_key = Alphanumeric.sample_string(&mut rng, PUBLIC_ID_LENGTH);
    let long_key = Alphanumeric.sample_string(&mut rng, LONG_KEY_LENGTH);
    let key = format!("{SK_PREFIX}-{T0_PREFIX}-{short_key}-{long_key}");
    SecretString::from(key)
}

impl TensorZeroApiKey {
    /// Hashes the long key using SHA-256, producing a hex string that we'll use for a database lookup
    fn hash_long_key(long_key: &str) -> String {
        hex::encode(Sha256::digest(long_key.as_bytes()))
    }

    #[cfg(feature = "e2e_tests")]
    pub fn new_for_testing(public_id: String, hashed_long_key: String) -> Self {
        Self {
            public_id,
            hashed_long_key: SecretString::from(hashed_long_key),
        }
    }

    pub fn get_public_id(&self) -> &str {
        &self.public_id
    }

    #[cfg(feature = "e2e_tests")]
    pub fn get_hashed_long_key(&self) -> SecretString {
        self.hashed_long_key.clone()
    }

    /// Validates that the provided key is of the format `sk-t0-<public_id>-<long_key>`,
    /// where <public_id> is 12 alphanumeric characters and <long_key> is 48 alphanumeric characters.
    /// Returns a `TensorZeroApiKey` containing the extracted public ID and long key.
    pub fn parse(key: &str) -> Result<Self, TensorZeroAuthError> {
        let parts = key.split('-').collect::<Vec<&str>>();
        let [sk, t0, public_id, long_key] = parts.as_slice() else {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must be of the form `sk-t0-<public_id>-<long_key>`",
            ));
        };
        if sk != &SK_PREFIX {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must start with `sk-t0-`",
            ));
        }
        if t0 != &T0_PREFIX {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "API key must start with `sk-t0-`",
            ));
        }
        if public_id.len() != PUBLIC_ID_LENGTH {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Public ID must be 12 characters",
            ));
        }
        if long_key.len() != LONG_KEY_LENGTH {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Long key must be 48 characters",
            ));
        }
        if !public_id.chars().all(char::is_alphanumeric) {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Public ID must be alphanumeric",
            ));
        }
        if !long_key.chars().all(char::is_alphanumeric) {
            return Err(TensorZeroAuthError::InvalidKeyFormat(
                "Long key must be alphanumeric",
            ));
        }
        Ok(Self {
            public_id: (*public_id).to_owned(),
            hashed_long_key: Self::hash_long_key(long_key).into(),
        })
    }

    /// Synapse keys are `sk-syn-v1-` plus 48 lowercase alphanumeric characters.
    pub fn is_synapse_key(key: &str) -> bool {
        let Some(rest) = key.strip_prefix(SYNAPSE_KEY_PREFIX) else {
            return false;
        };
        rest.len() == LONG_KEY_LENGTH
            && rest
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    }

    /// Identity used for request extensions / tracing when a Synapse key authenticates.
    /// The SHA-256 is only a cache key; verification uses bcrypt against the import table.
    pub fn from_synapse_plaintext(key: &str) -> Self {
        let hash = hex::encode(Sha256::digest(key.as_bytes()));
        let public_id = format!("syn{}", &hash[..9]);
        Self {
            public_id,
            hashed_long_key: hash.into(),
        }
    }

    /// Returns a cache key that includes both the public_id and the hashed long key.
    /// This ensures that cache entries are unique per full API key, not just per public_id.
    /// This is critical for security - using only the public_id would allow an attacker
    /// to bypass authentication by crafting a key with the same public_id but different secret.
    ///
    /// TODO: This is `pub` while we run the cache from the gateway but later we should internalize it.
    pub fn cache_key(&self) -> String {
        let TensorZeroApiKey {
            public_id,
            hashed_long_key,
        } = self;

        format!("{public_id}:{}", hashed_long_key.expose_secret())
    }
}

#[derive(Clone, Debug, Error)]
pub enum TensorZeroAuthError {
    #[error("Invalid format for TensorZero API key: {0}")]
    InvalidKeyFormat(&'static str),
    #[error("Invalid `expires_at` value: {0}")]
    InvalidExpiresAt(String),
    #[error("`workspace` filter requires `organization` filter to be set")]
    WorkspaceFilterRequiresOrganization,
    #[error("Database error: {0}")]
    Sqlx(Arc<sqlx::Error>),
    #[error("Migration error: {message}")]
    Migration { message: String },
    #[error("Error performing authentication: {message}")]
    Middleware {
        message: String,
        key_info: Option<Box<KeyInfo>>,
    },
}

impl From<sqlx::Error> for TensorZeroAuthError {
    fn from(err: sqlx::Error) -> Self {
        Self::Sqlx(Arc::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_invalid_key() {
        assert_eq!(
            TensorZeroApiKey::parse("invalid-key")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must be of the form `sk-t0-<public_id>-<long_key>`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("too-many-dashes-in-my-api-key")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must be of the form `sk-t0-<public_id>-<long_key>`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("bad-with-four-components")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must start with `sk-t0-`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-other-123456789012-12345678901234567890123456789012")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: API key must start with `sk-t0-`"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-t0-12-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: Public ID must be 12 characters"
        );
        assert_eq!(
            TensorZeroApiKey::parse("sk-t0-aaaaaaaaaaaa-bb")
                .unwrap_err()
                .to_string(),
            "Invalid format for TensorZero API key: Long key must be 48 characters"
        );
        assert_eq!(
            TensorZeroApiKey::parse(
                "sk-t0-!!!!!!!!!!!!-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
            .unwrap_err()
            .to_string(),
            "Invalid format for TensorZero API key: Public ID must be alphanumeric"
        );
        assert_eq!(
            TensorZeroApiKey::parse(
                "sk-t0-aaaaaaaaaaaa-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!"
            )
            .unwrap_err()
            .to_string(),
            "Invalid format for TensorZero API key: Long key must be alphanumeric"
        );
    }

    #[test]
    fn test_parse_valid_key() {
        assert_eq!(
            format!(
                "{:?}",
                TensorZeroApiKey::parse(
                    "sk-t0-123456789012-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                )
                .unwrap()
            ),
            "TensorZeroApiKey { public_id: \"123456789012\", hashed_long_key: SecretBox<str>([REDACTED]) }"
        );
    }

    #[test]
    fn test_synapse_key_format() {
        let key = format!("sk-syn-v1-{}", "a".repeat(48));
        assert!(TensorZeroApiKey::is_synapse_key(&key));
        assert!(!TensorZeroApiKey::is_synapse_key(
            "sk-t0-123456789012-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ));
        assert!(!TensorZeroApiKey::is_synapse_key(&format!(
            "sk-syn-v1-{}",
            "A".repeat(48)
        )));
        let parsed = TensorZeroApiKey::from_synapse_plaintext(&key);
        assert_eq!(parsed.public_id.len(), PUBLIC_ID_LENGTH);
        assert!(parsed.public_id.starts_with("syn"));
    }

    #[test]
    fn test_synapse_public_id_is_hash_of_plaintext_not_bcrypt() {
        let key = format!("sk-syn-v1-{}", "a".repeat(48));
        let from_plaintext = TensorZeroApiKey::from_synapse_plaintext(&key);
        let bcrypt_hash = "$2b$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy";
        let from_bcrypt = format!(
            "syn{}",
            &hex::encode(Sha256::digest(bcrypt_hash.as_bytes()))[..9]
        );
        assert_ne!(from_plaintext.public_id, from_bcrypt);
        assert_eq!(from_plaintext.public_id.len(), PUBLIC_ID_LENGTH);
    }
}
