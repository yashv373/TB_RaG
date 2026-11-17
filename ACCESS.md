# Access: NishkamTB_RaG on GitHub Pages

## What is public and what is not
GitHub Pages is public. So the library is **encrypted before it is published** (AES-256-GCM). The key comes from a
team passphrase (PBKDF2-SHA256, 600,000 rounds, random salt in the manifest). What anyone can see: the page code,
`index/manifest.json` (version, salt, model name, opaque shard names and hashes) and ciphertext. Book titles, authors,
passages, metadata, the search data and the vectors are all inside ciphertext (book titles and the shard-to-book map
sit in an encrypted `catalog-*.bin`). Shard folders use keyed opaque ids, so they cannot be matched to a book without the passphrase. Each shard is tied to its file name, so shards cannot be swapped.

## Publish
1. Build (passphrase only from the environment, never in a file or log):
   `RAG_PASSPHRASE=... python tools/build_index.py --add tools/<book>.json`
   Plaintext stays in `rag/build/` (gitignored). Only `rag/web/` is published.
2. Put `rag/web/` in its own repo, e.g. `yashv373/NishkamTB_RaG` (public is fine: ciphertext only; a private repo
   also works if your plan allows Pages on private repos). Settings > Pages > deploy from branch, folder `/`.
3. Open `https://yashv373.github.io/NishkamTB_RaG/`, enter the passphrase.

## Add a book
Run the same command with the new book file. Only that book's shards and the small `global` stats file change,
the version number goes up, and browsers fetch only the new shard files (names contain their hash).
Same passphrase and salt are reused, so old shards stay valid.

## Rotate the passphrase
`RAG_PASSPHRASE=<new> python tools/build_index.py --rotate` (new salt, every shard re-encrypted), then push.
Do this when someone leaves the team. Old copies of the ciphertext stay readable with the old passphrase.

## Threat model (plain words)
- Anyone with the passphrase can read every passage. Share it only with the team.
- Anyone without it sees only ciphertext. A weak passphrase can be guessed offline (600k rounds slows this but does
  not stop it): use a long random one (5+ random words).
- The page code itself is public and could be swapped by someone who controls the repo; do not type the passphrase
  into a copy hosted elsewhere.
- Once unlocked, text is in browser memory and sent to your chosen AI provider as part of the prompt. That is a
  copy of the passages going to that provider. Check its terms before using real book text.
- The chat history in IndexedDB keeps the answers and the source passages in clear text, so clear it on a shared machine.
- All `*.github.io/<repo>` sites share one browser origin. Do not put other sites on this account's Pages, since they could read the browser storage.
- API keys sit in localStorage (per provider, with a Forget button). The passphrase is kept only in sessionStorage
  if you tick "remember", and the Lock button clears it. Chat history is in IndexedDB on that machine.
- The library is a licensed, copyrighted textbook set. It is encrypted, but do not mirror the repo outside the team.
