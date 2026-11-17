// WebCrypto side of the index encryption. File = iv(12) || ciphertext || tag, AAD = file name.
const te = new TextEncoder();
const hex = (h) => Uint8Array.from(h.match(/../g).map((x) => parseInt(x, 16)));
export async function deriveKey(pass, kdf) {
  const km = await crypto.subtle.importKey('raw', te.encode(pass), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey({ name: 'PBKDF2', salt: hex(kdf.salt), iterations: kdf.iterations, hash: 'SHA-256' },
    km, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
}
export async function decrypt(key, name, buf) { // throws on wrong key or swapped file
  const b = new Uint8Array(buf);
  return crypto.subtle.decrypt({ name: 'AES-GCM', iv: b.slice(0, 12), additionalData: te.encode(name) }, key, b.slice(12));
}
export async function checkPassphrase(key, name, buf) {
  try { return new TextDecoder().decode(await decrypt(key, name, buf)) === 'nishkam-rag-check-v1'; } catch { return false; }
}
