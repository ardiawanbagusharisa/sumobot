# Hooking the login system up to the leaderboard

This is for whoever is building login and account-side leaderboard saving. Short version: you don't need to touch any leaderboard UI code. There are four places where your system and the leaderboard meet, and the first one is a single method call.

## 1. Tell the leaderboard who the player is

The leaderboard tracks players by a `ProfileID` string. Before anyone logs in, that ID is an anonymous GUID the game generates on first launch and keeps in PlayerPrefs. Your job is simply to replace it with your account ID once the player signs in:

```csharp
// right after a successful login:
GameManager.Instance.ApplyAccount(account.UserId, account.Username);
```

That one call does three things: switches the player's identity to your server-issued ID, saves it locally so it survives restarts, and moves any ratings the player earned while anonymous over to the account. If the account already had ratings on some board, the account's record wins and the anonymous duplicate is thrown away.

Call it early — ideally before the player starts a battle — so results get recorded under the right identity. Calling it again on later launches is harmless.

One rule for your registration side: account IDs must never start with `bot:`. That prefix is reserved for AI bots on the boards, and both `ApplyAccount` and the leaderboard service will refuse IDs that use it.

## 2. Saving leaderboard data to your server

All leaderboard data lives in one plain JSON object (`LeaderboardData`) and everything that reads or writes it goes through one small interface:

```csharp
public interface ILeaderboardStore
{
    LeaderboardData Load();
    void Save(LeaderboardData data);
}
```

The built-in `LeaderboardStore` implements this by writing `Leaderboards/leaderboards.json` under `Application.persistentDataPath`. You have two ways in, from easiest to most involved:

**Easy way — piggyback on the local file.** Keep the default store. After login, upload the JSON file to your backend as the account's cloud save; on login, download it and put it back before the game reads it. Zero code changes on the leaderboard side.

**Proper way — bring your own store.** Write a class implementing `ILeaderboardStore` that talks to your backend (or wraps the local store and syncs in the background), then plug it in:

```csharp
// after login:
LeaderboardService.Instance.SetStore(new MyAccountBackedStore(session), reload: true);

// or, if your store can exist before the service is first used:
LeaderboardService.StoreFactory = () => new MyAccountBackedStore(session);
```

`SetStore` swaps the backend at runtime and reloads; `StoreFactory` makes the service start on your store from the beginning.

One warning if your server modifies the data: the local file carries a checksum, and the client resets any file whose checksum doesn't match. If your store returns server data, either don't reuse the local file format's checksum field, or recompute it with `LeaderboardStore.ComputeChecksum` before handing data back.

## 3. Where results come from

You never need to compute ratings yourself. Every finished battle funnels through exactly one method — `LeaderboardService.RecordBattle(...)` — which is called once per match by `BattleManager` when the result screen appears. It applies Elo, saves through whatever store is active, and then fires the service's `Updated` event. If you want to sync to the server after every match, subscribe to `Updated` — it's your "something changed, push now" signal.

If you ever go fully server-authoritative (server computes ratings), this is the single call site you'd forward to the backend.

## 4. What reads the data

The leaderboard screen only ever calls `GetTable`, `GetRank` and `GetEntry` on the service. As long as your store returns a valid `LeaderboardData`, the whole UI — filters, rankings, the player's rank chip, rename — works unchanged no matter where the data actually came from.

## Small print

- Only Player 1 is treated as "the local account". Player 2 in hot-seat stays a local device profile.
- Bots appear on the boards as their own entities with `bot:<BotID>` IDs — leave those entries alone when syncing accounts.
- Renaming: the game already updates board entries when a player renames themselves (`LeaderboardService.RenameProfile`). If usernames come from your server, pass the fresh name through `ApplyAccount` and it's handled.
- Batch simulations (`BattleSimulator`) never write to these boards by design, so you won't get junk data from test runs.

Questions about any of this: the relevant code is all under `Assets/Scripts/Leaderboard/` plus `ApplyAccount` in `GameManager.cs`, and the bigger roadmap (including the planned online Phase 2) is in `Wiki/Leaderboard-Development-Plan.md`.
