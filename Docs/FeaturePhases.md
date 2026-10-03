# Feature phase status

The current branch provides a complete local implementation that can be tested without paid services. Visual Scripting remains intentionally deferred.

## Phase 1 — dummy login

- Automatic anonymous local login with an editable display name.
- Stable player ID and JSON save data across restarts.
- Launch profiles (`sumobot_p1`, `sumobot_p2`) use separate dummy accounts, inventories, and names on the same machine.

No additional account is required. The Unity ID and linked Unity Cloud project used by online matchmaking are sufficient.

## Phase 2 — inventory and equipment

- Owned catalog item IDs and equipped items are persisted per account.
- Buying an equippable item changes the primary action to **Equip**.
- The equipped state changes to **Equipped** and is applied to the robot in battle.
- Equipped wheel/accessory sprites and tints are synchronized to both online peers.

Items without an equipment slot (AI scripts, modules, and the deferred visual-script content) remain ownable but do not show an Equip action.

## Phase 3 — market

- Runtime catalog loaded from `Resources/Catalog/items.json`.
- Coin balance, validation, purchase persistence, duplicate protection, and refund-on-grant-failure.
- Successful purchases immediately refresh the inventory and balance views.

## Phase 4 — online PvP

- Unity Authentication profiles, Multiplayer Services matchmaking, Relay over DTLS, Unity Transport, and Netcode for GameObjects.
- Host-authoritative battle simulation with synchronized player input, names, loadouts, match state, and rematch consent.

See `Docs/Multiplayer.md` for build and two-instance testing.

## Phase 5 — leaderboard

- Elo tables, match statistics, filtering, persistent identities, and result UI.
- Local, campaign, and online-PvP results use the same result pipeline.
- Online results are recorded independently on both peers, making same-machine and LAN/Internet tests work without configuring a leaderboard service.

This phase currently uses local JSON persistence. A global cheat-resistant leaderboard requires a trusted dedicated server or Cloud Code validation; a client-hosted peer cannot provide authoritative anti-cheat guarantees by itself.
