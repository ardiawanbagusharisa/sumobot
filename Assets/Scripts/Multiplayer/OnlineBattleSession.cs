using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SumoCore;
using SumoInput;
using SumoManager;
using SumoServices;
using TMPro;
using Unity.Collections;
using Unity.Netcode;
using Unity.Netcode.Transports.UTP;
using Unity.Services.Authentication;
using Unity.Services.Core;
using Unity.Services.Core.Environments;
using Unity.Services.Multiplayer;
using Unity.Services.Relay.Models;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace SumoMultiplayer
{
    /// <summary>
    /// Owns the complete online-match lifecycle for the two-player prototype:
    /// UGS anonymous authentication, selectable session rooms, Relay/DTLS, NGO scene
    /// loading, host-authoritative input, and client battle snapshots.
    ///
    /// This component is created before the first scene, so no scene YAML needs
    /// package-specific references. It intentionally supports one host and one
    /// joining client; a dedicated server can replace this boundary later.
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed class OnlineBattleSession : MonoBehaviour
    {
        private const string MatchType = "sumobot-online-v2";
        private const string MatchModeProperty = "mode";
        private const string DisplayNameProperty = "displayName";
        private const string ReadyMessage = "sumobot/ready/v1";
        private const string LobbyReadyMessage = "sumobot/lobby-ready/v1";
        private const string LobbyStateMessage = "sumobot/lobby-state/v1";
        private const string BattleReadyMessage = "sumobot/battle-ready/v1";
        private const string MatchInfoMessage = "sumobot/match-info/v1";
        private const string InputSelectionMessage = "sumobot/input-selection/v1";
        private const string InputMessage = "sumobot/input/v1";
        private const string SnapshotMessage = "sumobot/snapshot/v2";
        private const string VfxMessage = "sumobot/vfx/v1";
        private const string RematchRequestMessage = "sumobot/rematch/v1";
        private const float SnapshotInterval = 1f / 20f;
        private const float ContinuousInputInterval = 0.075f;
        private const float RematchWindowSeconds = 15f;
        private const float LobbyReadySeconds = 30f;
        private const float BattlePreparationSeconds = 10f;
        private const int OnlineDisconnectTimeoutMs = 5000;
        private const byte DashVfx = 1;
        private const byte CollisionVfx = 2;
        private const byte AcceleratingFlag = 1;
        private const byte TurnLeftFlag = 2;
        private const byte TurnRightFlag = 4;
        private const byte DashActiveFlag = 8;
        private const byte SkillActiveFlag = 16;

        public static OnlineBattleSession Instance { get; private set; }
        public static bool IsActive => Instance != null && Instance.onlineMatchActive;
        public static bool IsHost => IsActive && Instance.session != null && Instance.session.IsHost;
        public static bool IsClient => IsActive && !IsHost;
        public static PlayerSide LocalSide => IsHost ? PlayerSide.Left : PlayerSide.Right;
        public static string ProfileName => Instance != null ? Instance.profileName : OnlineLaunchOptions.GetProfileName();
        public static string LeftDisplayName => Instance != null ? Instance.leftDisplayName : "Player 1";
        public static string RightDisplayName => Instance != null ? Instance.rightDisplayName : "Player 2";

        private NetworkManager networkManager;
        private UnityTransport transport;
        private ISession session;
        private Button onlineButton;
        private TMP_Text onlineStatus;
        private OnlineRoomPanel roomPanel;
        private BattleManager boundBattleManager;
        private Button rematchButton;
        private TMP_Text rematchLabel;
        private Button startBattleButton;
        private TMP_Text startBattleLabel;

        private readonly Dictionary<ActionType, float> lastInputSentAt = new();
        private SnapshotTarget leftTarget;
        private SnapshotTarget rightTarget;
        private string profileName;
        private string matchPool;
        private string localDisplayName;
        private string localAccountId;
        private string localEquipment;
        private string leftDisplayName = "Player 1";
        private string rightDisplayName = "Player 2";
        private string leftAccountId = "player:left";
        private string rightAccountId = "player:right";
        private string leftEquipment = string.Empty;
        private string rightEquipment = string.Empty;
        private bool autoCreateRoom;
        private bool autoJoinRoom;
        private bool autoLobbyReady;
        private bool matchmaking;
        private bool onlineMatchActive;
        private bool messagesRegistered;
        private bool localReadySent;
        private bool battleSceneRequested;
        private bool leaving;
        private bool clientReady;
        private bool lobbyCountdownActive;
        private bool leftLobbyReady;
        private bool rightLobbyReady;
        private float lobbyDeadline;
        private float clientLobbyRemaining;
        private float clientLobbyStateReceivedAt;
        private float nextLobbyStateAt;
        private float nextSnapshotAt;
        private bool rematchWindowActive;
        private bool leftRematchRequested;
        private bool rightRematchRequested;
        private float rematchDeadline;
        private float clientRematchRemaining;
        private float rematchSnapshotReceivedAt;
        private InputType localInputType = InputType.UI;
        private InputType leftInputType = InputType.UI;
        private InputType rightInputType = InputType.UI;
        private bool refreshingRooms;
        private float nextRoomRefreshAt;
        private bool connectionLossHandled;
        private bool inputSelectionOpen;
        private bool hostBattleReady;
        private bool clientBattleReady;
        private bool localBattleReady;
        private bool localBattleReadySent;
        private bool startRequested;
        private bool preparationActive;
        private bool onlineStartAuthorized;
        private float preparationDeadline;
        private float clientPreparationRemaining;
        private float clientPreparationReceivedAt;

        public static bool IsHostStartAuthorized => IsHost && Instance.onlineStartAuthorized;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Bootstrap()
        {
            if (Instance != null)
                return;

            var root = new GameObject("OnlineBattleSession");
            DontDestroyOnLoad(root);
            root.AddComponent<OnlineBattleSession>();
        }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }

            Instance = this;
            profileName = OnlineLaunchOptions.GetProfileName();
            localDisplayName = profileName;
            localAccountId = profileName;
            localEquipment = string.Empty;
            matchPool = OnlineLaunchOptions.GetMatchPoolName();
            autoCreateRoom = OnlineLaunchOptions.HasAutoCreateRoomFlag();
            autoJoinRoom = OnlineLaunchOptions.HasAutoJoinRoomFlag();
            if (OnlineLaunchOptions.HasAutoOnlineFlag())
            {
                autoCreateRoom = profileName.EndsWith("_p1", StringComparison.OrdinalIgnoreCase);
                autoJoinRoom = !autoCreateRoom;
            }
            autoLobbyReady = autoCreateRoom || autoJoinRoom;
            Application.runInBackground = true;

            transport = gameObject.AddComponent<UnityTransport>();
            // UTP defaults to 30 seconds, which leaves the remaining player in a
            // dead match after a crash or force-close. Five seconds is responsive
            // enough for a forfeit while still tolerating short internet jitter.
            transport.DisconnectTimeoutMS = OnlineDisconnectTimeoutMs;
            networkManager = gameObject.AddComponent<NetworkManager>();
            // NetworkConfig is serialized when NetworkManager lives in a scene,
            // but NGO leaves it null when the component is created at runtime.
            networkManager.NetworkConfig = new NetworkConfig();
            networkManager.NetworkConfig.NetworkTransport = transport;
            networkManager.NetworkConfig.EnableSceneManagement = true;
            // Snapshot v2 includes client-only presentation state. Refuse older
            // builds instead of decoding their shorter snapshot payload.
            networkManager.NetworkConfig.ProtocolVersion = 2;

            SceneManager.sceneLoaded += OnSceneLoaded;
            networkManager.OnClientConnectedCallback += OnClientConnected;
            networkManager.OnClientDisconnectCallback += OnClientDisconnected;
        }

        private void Start()
        {
            // RuntimeInitializeOnLoad creates this object before the first scene, but
            // Unity can complete the initial MainMenu load before the sceneLoaded
            // subscription observes it on slower client startups. Bind the active
            // scene as an idempotent fallback so command-line auto create/join flags
            // and the room browser are never skipped.
            if (SceneManager.GetActiveScene().name == "MainMenu")
                BindOnlineButton();
        }

        private void OnDestroy()
        {
            if (Instance != this)
                return;

            SceneManager.sceneLoaded -= OnSceneLoaded;
            if (networkManager != null)
            {
                networkManager.OnClientConnectedCallback -= OnClientConnected;
                networkManager.OnClientDisconnectCallback -= OnClientDisconnected;
            }

            UnbindBattleManager();
            UnregisterMessages();
            Instance = null;
        }

        private void OnApplicationQuit()
        {
            leaving = true;
        }

        private void Update()
        {
            if (roomPanel != null && roomPanel.IsVisible &&
                !onlineMatchActive && !refreshingRooms && !matchmaking &&
                Time.unscaledTime >= nextRoomRefreshAt)
            {
                nextRoomRefreshAt = Time.unscaledTime + 5f;
                _ = RefreshRoomsAsync();
            }

            if (onlineMatchActive)
                TryCompleteNetworkSetup();

            if (onlineMatchActive && !battleSceneRequested &&
                SceneManager.GetActiveScene().name == "MainMenu")
            {
                if (IsHost && lobbyCountdownActive)
                {
                    if (leftLobbyReady && rightLobbyReady || GetLobbyRemaining() <= 0f)
                        BeginOnlineBattle();
                    else if (Time.unscaledTime >= nextLobbyStateAt)
                    {
                        nextLobbyStateAt = Time.unscaledTime + 1f;
                        SendLobbyState();
                    }
                }
                RefreshLobbyUI();
            }

            if (!onlineMatchActive || SceneManager.GetActiveScene().name != "Battle")
                return;

            if (IsHost)
            {
                if (preparationActive)
                {
                    if (boundBattleManager == null ||
                        boundBattleManager.CurrentState != BattleState.PreBatle_Preparing)
                        preparationActive = false;
                    else if (GetPreparationRemaining() <= 0f)
                    {
                        preparationActive = false;
                        StartAuthorizedBattle();
                    }
                }

                if (rematchWindowActive && GetRematchRemaining() <= 0f)
                    rematchWindowActive = false;

                if (Time.unscaledTime >= nextSnapshotAt)
                {
                    nextSnapshotAt = Time.unscaledTime + SnapshotInterval;
                    SendBattleSnapshot();
                }
            }
            else
            {
                if (localBattleReady && !localBattleReadySent)
                    SendClientBattleReady();
                InterpolateClientView();
            }

            RefreshStartBattleUI();

            if (boundBattleManager != null &&
                boundBattleManager.CurrentState == BattleState.PostBattle_ShowResult)
            {
                RefreshRematchUI();
            }
        }

        private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (scene.name == "MainMenu")
            {
                UnbindBattleManager();
                BindOnlineButton();
                return;
            }

            if (scene.name == "Battle" && onlineMatchActive)
            {
                if (IsClient)
                    ResetInitialStartState();
                StartCoroutine(AttachBattleWhenReady());
            }
        }

        private void BindOnlineButton()
        {
            onlineButton = Resources.FindObjectsOfTypeAll<Button>()
                .FirstOrDefault(candidate =>
                    candidate.name == "ButtonOnline" && candidate.gameObject.scene.IsValid());

            if (onlineButton == null)
            {
                Logger.Error("[Online] ButtonOnline was not found in MainMenu.");
                return;
            }

            TMP_Text[] labels = onlineButton.GetComponentsInChildren<TMP_Text>(true);
            onlineStatus = labels.FirstOrDefault(label =>
                label.text.IndexOf("coming", StringComparison.OrdinalIgnoreCase) >= 0)
                ?? labels.LastOrDefault();

            onlineButton.interactable = true;
            onlineButton.onClick.RemoveAllListeners();
            onlineButton.onClick.AddListener(OnOnlineButtonPressed);

            if (!matchmaking && !onlineMatchActive)
                SetStatus($"Browse rooms\nProfile: {profileName}");

            if ((autoCreateRoom || autoJoinRoom) && !matchmaking && !onlineMatchActive)
            {
                bool create = autoCreateRoom;
                autoCreateRoom = false;
                autoJoinRoom = false;
                _ = RunAutomatedRoomFlowAsync(create);
            }
        }

        private void OnOnlineButtonPressed()
        {
            if (onlineMatchActive)
            {
                if (SceneManager.GetActiveScene().name == "MainMenu")
                    _ = CancelWaitingSessionAsync();

                return;
            }

            _ = ShowRoomBrowserAsync();
        }

        private async Task CancelWaitingSessionAsync()
        {
            if (leaving)
                return;

            SetStatus("Leaving online session...");
            await LeaveSessionAsync(true);
            roomPanel?.SetWaiting(false);
            roomPanel?.SetStatus("Room closed.");
            SetStatus($"Browse rooms\nProfile: {profileName}");
        }

        private async Task ShowRoomBrowserAsync()
        {
            EnsureRoomPanel();
            roomPanel.Show(localInputType);
            roomPanel.SetWaiting(false);
            roomPanel.SetStatus("Signing in and loading rooms...");
            await RefreshRoomsAsync();
        }

        private void EnsureRoomPanel()
        {
            if (roomPanel != null)
                return;

            Canvas canvas = onlineButton != null ? onlineButton.GetComponentInParent<Canvas>() : null;
            if (canvas == null)
                throw new InvalidOperationException("The online room browser requires a parent Canvas.");

            roomPanel = OnlineRoomPanel.Create(
                canvas,
                onlineStatus != null ? onlineStatus.font : null,
                () => _ = CreateRoomAsync(),
                () => _ = RefreshRoomsAsync(),
                OnRoomPanelCloseRequested,
                RequestLobbyReady,
                roomId => _ = JoinRoomAsync(roomId));
        }

        private void OnRoomPanelCloseRequested()
        {
            if (onlineMatchActive)
            {
                _ = CancelWaitingSessionAsync();
                return;
            }

            roomPanel?.Hide();
        }

        private async Task RefreshRoomsAsync()
        {
            if (refreshingRooms || onlineMatchActive)
                return;

            refreshingRooms = true;
            roomPanel?.SetBusy(true);
            roomPanel?.SetStatus("Loading open rooms...");

            try
            {
                await InitializeUnityServicesAsync();
                RefreshLocalIdentity();

                QuerySessionsResults results = await MultiplayerService.Instance.QuerySessionsAsync(
                    new QuerySessionsOptions
                    {
                        Count = 20,
                        FilterOptions = new List<FilterOption>
                        {
                            new(FilterField.AvailableSlots, "1", FilterOperation.GreaterOrEqual),
                            new(FilterField.StringIndex1, matchPool, FilterOperation.Equal)
                        },
                        SortOptions = new List<SortOption>
                        {
                            new(SortOrder.Descending, SortField.CreationTime)
                        }
                    });

                List<OnlineRoomPanel.RoomEntry> rooms = results.Sessions
                    .Where(info => !info.IsLocked && !info.HasPassword && info.AvailableSlots > 0)
                    .Select(info => new OnlineRoomPanel.RoomEntry(
                        info.Id,
                        string.IsNullOrWhiteSpace(info.Name) ? "Sumobot room" : info.Name,
                        info.MaxPlayers - info.AvailableSlots,
                        info.MaxPlayers))
                    .ToList();

                roomPanel?.SetRooms(rooms);
                if (rooms.Count > 0)
                    roomPanel?.SetStatus("Choose a room, or create a new one.");
            }
            catch (Exception ex)
            {
                Logger.Error($"[Online] Failed to query rooms: {ex}");
                roomPanel?.SetStatus($"Could not load rooms: {FriendlyError(ex)}");
            }
            finally
            {
                refreshingRooms = false;
                roomPanel?.SetBusy(false);
                nextRoomRefreshAt = Time.unscaledTime + 5f;
            }
        }

        private async Task CreateRoomAsync()
        {
            if (matchmaking || onlineMatchActive)
                return;

            matchmaking = true;
            roomPanel?.SetBusy(true);
            roomPanel?.SetStatus("Creating room...");
            try
            {
                await PrepareSessionOperationAsync();
                localInputType = roomPanel?.SelectedInputType ?? InputType.UI;

                string roomName = $"{localDisplayName}'s room";
                if (roomName.Length > 60)
                    roomName = roomName.Substring(0, 60);

                var options = new SessionOptions
                {
                    Name = roomName,
                    Type = MatchType,
                    MaxPlayers = 2,
                    IsPrivate = false,
                    PlayerProperties = CreatePlayerProperties(),
                    SessionProperties = CreateSessionProperties()
                }
                .WithRelayNetwork()
                .WithNetworkOptions(new NetworkOptions { RelayProtocol = RelayProtocol.DTLS });

                ISession createdSession = await MultiplayerService.Instance.CreateSessionAsync(options);
                AdoptSession(createdSession);
                roomPanel?.SetWaiting(true);
                roomPanel?.SetStatus($"Room: {createdSession.Name}\nWaiting for an opponent...");
                SetStatus("Waiting in room...\nClick to leave");
            }
            catch (Exception ex)
            {
                HandleRoomOperationFailure("create room", ex);
            }
            finally
            {
                matchmaking = false;
                if (!onlineMatchActive)
                    roomPanel?.SetBusy(false);
            }
        }

        private async Task JoinRoomAsync(string roomId)
        {
            if (matchmaking || onlineMatchActive || string.IsNullOrWhiteSpace(roomId))
                return;

            matchmaking = true;
            roomPanel?.SetBusy(true);
            roomPanel?.SetStatus("Joining room...");
            try
            {
                await PrepareSessionOperationAsync();
                localInputType = roomPanel?.SelectedInputType ?? InputType.UI;

                var options = new JoinSessionOptions
                {
                    Type = MatchType,
                    PlayerProperties = CreatePlayerProperties()
                }
                .WithNetworkOptions(new NetworkOptions { RelayProtocol = RelayProtocol.DTLS });

                ISession joinedSession = await MultiplayerService.Instance.JoinSessionByIdAsync(roomId, options);
                AdoptSession(joinedSession);
                roomPanel?.SetWaiting(true);
                roomPanel?.SetStatus("Connected. Waiting for the host's ready timer...");
                SetStatus("Connecting to room...");
                TryCompleteNetworkSetup();
            }
            catch (Exception ex)
            {
                HandleRoomOperationFailure("join room", ex);
                await RefreshRoomsAsync();
            }
            finally
            {
                matchmaking = false;
                if (!onlineMatchActive)
                    roomPanel?.SetBusy(false);
            }
        }

        private async Task PrepareSessionOperationAsync()
        {
            localReadySent = false;
            connectionLossHandled = false;
            ResetLobbyState();
            await EnsureNetworkManagerStoppedAsync();
            await InitializeUnityServicesAsync();
            RefreshLocalIdentity();
        }

        private Dictionary<string, PlayerProperty> CreatePlayerProperties()
        {
            return new Dictionary<string, PlayerProperty>
            {
                [DisplayNameProperty] = new(localDisplayName, VisibilityPropertyOptions.Member)
            };
        }

        private Dictionary<string, SessionProperty> CreateSessionProperties()
        {
            return new Dictionary<string, SessionProperty>
            {
                [MatchModeProperty] = new(
                    matchPool,
                    VisibilityPropertyOptions.Public,
                    PropertyIndex.String1)
            };
        }

        private void AdoptSession(ISession joinedSession)
        {
            session = joinedSession ?? throw new InvalidOperationException("UGS returned no session.");
            onlineMatchActive = true;
            connectionLossHandled = false;
            if (session.IsHost)
            {
                leftDisplayName = localDisplayName;
                leftAccountId = localAccountId;
                leftEquipment = localEquipment;
                leftInputType = localInputType;
            }
            else
            {
                rightDisplayName = localDisplayName;
                rightAccountId = localAccountId;
                rightEquipment = localEquipment;
                rightInputType = localInputType;
            }

            SubscribeSessionEvents();
            RegisterMessages();
            Logger.Info(
                $"[Online] {(session.IsHost ? "Created" : "Joined")} room {session.Id} " +
                $"with profile {profileName}, input {localInputType}, pool {matchPool}.");
        }

        private void HandleRoomOperationFailure(string operation, Exception ex)
        {
            onlineMatchActive = false;
            session = null;
            Logger.Error($"[Online] Failed to {operation}: {ex}");
            roomPanel?.SetWaiting(false);
            roomPanel?.SetStatus($"Could not {operation}: {FriendlyError(ex)}");
            SetStatus($"Online unavailable\n{FriendlyError(ex)}");
        }

        private async Task RunAutomatedRoomFlowAsync(bool create)
        {
            await ShowRoomBrowserAsync();
            if (create)
            {
                await CreateRoomAsync();
                return;
            }

            // Session query results can take several seconds to become visible
            // after the host creates a room. Keep the smoke-test joiner alive
            // long enough for that propagation instead of silently giving up.
            for (int attempt = 0; attempt < 60 && !onlineMatchActive; attempt++)
            {
                try
                {
                    await InitializeUnityServicesAsync();
                    QuerySessionsResults results = await MultiplayerService.Instance.QuerySessionsAsync(
                        new QuerySessionsOptions
                        {
                            Count = 1,
                            FilterOptions = new List<FilterOption>
                            {
                                new(FilterField.AvailableSlots, "1", FilterOperation.GreaterOrEqual),
                                new(FilterField.StringIndex1, matchPool, FilterOperation.Equal)
                            }
                        });
                    ISessionInfo room = results.Sessions.FirstOrDefault();
                    if (room != null)
                    {
                        await JoinRoomAsync(room.Id);
                        if (onlineMatchActive)
                            return;
                    }
                }
                catch (Exception ex)
                {
                    Logger.Warning($"[Online] Automated room query attempt {attempt + 1} failed: {FriendlyError(ex)}");
                }

                await Task.Delay(500);
            }

            Logger.Warning($"[Online] Automated test could not find a host room in pool {matchPool}.");
            roomPanel?.SetStatus("Automated test could not find the host room.");
        }

        private async Task InitializeUnityServicesAsync()
        {
            if (UnityServices.State == ServicesInitializationState.Uninitialized)
            {
                var options = new InitializationOptions();
                options.SetProfile(profileName);

                string environment = OnlineLaunchOptions.GetEnvironmentName();
                if (!string.IsNullOrWhiteSpace(environment))
                    options.SetEnvironmentName(environment);

                await UnityServices.InitializeAsync(options);
            }

            if (!AuthenticationService.Instance.IsSignedIn)
                await AuthenticationService.Instance.SignInAnonymouslyAsync();

            if (AuthenticationService.Instance.Profile != profileName)
            {
                throw new InvalidOperationException(
                    $"UGS was initialized with profile '{AuthenticationService.Instance.Profile}', " +
                    $"but this client requested '{profileName}'.");
            }
        }

        private void SubscribeSessionEvents()
        {
            if (session == null)
                return;

            session.PlayerJoined += OnSessionPlayerJoined;
            session.PlayerHasLeft += OnSessionPlayerLeft;
            session.RemovedFromSession += OnRemovedFromSession;
            session.Deleted += OnSessionDeleted;
        }

        private void UnsubscribeSessionEvents()
        {
            if (session == null)
                return;

            session.PlayerJoined -= OnSessionPlayerJoined;
            session.PlayerHasLeft -= OnSessionPlayerLeft;
            session.RemovedFromSession -= OnRemovedFromSession;
            session.Deleted -= OnSessionDeleted;
        }

        private void RegisterMessages()
        {
            if (messagesRegistered || networkManager == null || !networkManager.IsListening)
                return;

            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(ReadyMessage, OnReadyMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(LobbyReadyMessage, OnLobbyReadyMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(LobbyStateMessage, OnLobbyStateMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(BattleReadyMessage, OnBattleReadyMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(MatchInfoMessage, OnMatchInfoMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(InputSelectionMessage, OnInputSelectionMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(InputMessage, OnInputMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(SnapshotMessage, OnSnapshotMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(VfxMessage, OnVfxMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(RematchRequestMessage, OnRematchRequestMessage);
            messagesRegistered = true;
        }

        private void TryCompleteNetworkSetup()
        {
            if (!onlineMatchActive || networkManager == null)
                return;

            if (networkManager.IsListening)
                RegisterMessages();

            if (IsClient && networkManager.IsConnectedClient && messagesRegistered && !localReadySent)
            {
                localReadySent = true;
                SetStatus("Connected. Waiting in room...");
                SendClientReady();
            }
        }

        private void UnregisterMessages()
        {
            if (!messagesRegistered || networkManager == null || networkManager.CustomMessagingManager == null)
                return;

            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(ReadyMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(LobbyReadyMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(LobbyStateMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(BattleReadyMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(MatchInfoMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(InputSelectionMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(InputMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(SnapshotMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(VfxMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(RematchRequestMessage);
            messagesRegistered = false;
        }

        private void SendClientReady()
        {
            if (!IsClient || !networkManager.IsConnectedClient)
                return;

            using var writer = new FastBufferWriter(1024, Allocator.Temp);
            writer.WriteValueSafe((byte)1);
            writer.WriteValueSafe(new FixedString64Bytes(localDisplayName));
            writer.WriteValueSafe(new FixedString128Bytes(localAccountId));
            writer.WriteValueSafe(new FixedString512Bytes(localEquipment));
            writer.WriteValueSafe((byte)localInputType);
            networkManager.CustomMessagingManager.SendNamedMessage(
                ReadyMessage,
                NetworkManager.ServerClientId,
                writer,
                NetworkDelivery.ReliableSequenced);
        }

        private void OnClientConnected(ulong clientId)
        {
            if (!onlineMatchActive)
                return;

            Logger.Info($"[Online] NGO client connected: {clientId}.");
            TryCompleteNetworkSetup();
        }

        private void OnReadyMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || senderClientId == networkManager.LocalClientId)
                return;

            reader.ReadValueSafe(out byte ready);
            reader.ReadValueSafe(out FixedString64Bytes clientName);
            reader.ReadValueSafe(out FixedString128Bytes clientAccount);
            reader.ReadValueSafe(out FixedString512Bytes clientLoadout);
            reader.ReadValueSafe(out byte clientInputMode);
            clientReady = ready == 1;
            rightDisplayName = clientName.IsEmpty ? "Player 2" : clientName.ToString();
            rightAccountId = clientAccount.IsEmpty ? "player:right" : clientAccount.ToString();
            rightEquipment = clientLoadout.ToString();
            rightInputType = ToSupportedOnlineInput(clientInputMode);
            SendMatchInfo(senderClientId);
            if (clientReady)
                BeginLobbyCountdown();
        }

        private void BeginLobbyCountdown()
        {
            if (!IsHost || battleSceneRequested || session == null ||
                session.PlayerCount < 2 &&
                !(clientReady && networkManager != null &&
                  networkManager.ConnectedClientsIds.Count >= 2))
                return;

            if (lobbyCountdownActive)
            {
                if (clientReady)
                    SendLobbyState();
                return;
            }

            lobbyCountdownActive = true;
            lobbyDeadline = Time.unscaledTime + LobbyReadySeconds;
            nextLobbyStateAt = Time.unscaledTime + 1f;
            leftLobbyReady = autoLobbyReady;
            rightLobbyReady = false;
            roomPanel?.SetStatus("Opponent joined. Ready up or wait for the timer.");
            SetStatus("Opponent joined. Ready up in the room.");
            SendLobbyState();
            RefreshLobbyUI();
            Logger.Info($"[Online] Both players are in the room. Ready deadline: {LobbyReadySeconds:0}s.");
        }

        private void RequestLobbyReady()
        {
            if (!onlineMatchActive || !lobbyCountdownActive || battleSceneRequested ||
                SceneManager.GetActiveScene().name != "MainMenu")
                return;

            if (IsHost)
            {
                if (leftLobbyReady)
                    return;
                leftLobbyReady = true;
                SendLobbyState();
                if (rightLobbyReady)
                    BeginOnlineBattle();
            }
            else
            {
                if (rightLobbyReady || networkManager == null ||
                    !networkManager.IsConnectedClient || !messagesRegistered)
                    return;

                rightLobbyReady = true;
                using var writer = new FastBufferWriter(sizeof(byte), Allocator.Temp);
                writer.WriteValueSafe((byte)1);
                networkManager.CustomMessagingManager.SendNamedMessage(
                    LobbyReadyMessage,
                    NetworkManager.ServerClientId,
                    writer,
                    NetworkDelivery.ReliableSequenced);
            }

            RefreshLobbyUI();
        }

        private void OnLobbyReadyMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || !lobbyCountdownActive || battleSceneRequested ||
                senderClientId == networkManager.LocalClientId)
                return;

            reader.ReadValueSafe(out byte ready);
            if (ready != 1 || rightLobbyReady)
                return;

            rightLobbyReady = true;
            SendLobbyState();
            RefreshLobbyUI();
            if (leftLobbyReady)
                BeginOnlineBattle();
        }

        private void SendLobbyState()
        {
            if (!IsHost || !lobbyCountdownActive || networkManager == null ||
                !networkManager.IsListening || !messagesRegistered)
                return;

            using var writer = new FastBufferWriter(16, Allocator.Temp);
            writer.WriteValueSafe((byte)(leftLobbyReady ? 1 : 0));
            writer.WriteValueSafe((byte)(rightLobbyReady ? 1 : 0));
            writer.WriteValueSafe(GetLobbyRemaining());
            foreach (ulong clientId in networkManager.ConnectedClientsIds)
            {
                if (clientId != networkManager.LocalClientId)
                    networkManager.CustomMessagingManager.SendNamedMessage(
                        LobbyStateMessage, clientId, writer, NetworkDelivery.ReliableSequenced);
            }
        }

        private void OnLobbyStateMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsClient || senderClientId != NetworkManager.ServerClientId ||
                SceneManager.GetActiveScene().name != "MainMenu")
                return;

            reader.ReadValueSafe(out byte hostReady);
            reader.ReadValueSafe(out byte guestReady);
            reader.ReadValueSafe(out float remaining);
            lobbyCountdownActive = true;
            leftLobbyReady = hostReady == 1;
            rightLobbyReady = guestReady == 1;
            clientLobbyRemaining = Mathf.Clamp(remaining, 0f, LobbyReadySeconds);
            clientLobbyStateReceivedAt = Time.unscaledTime;
            roomPanel?.SetStatus("Opponent joined. Ready up or wait for the timer.");
            if (autoLobbyReady && !rightLobbyReady)
                RequestLobbyReady();
            RefreshLobbyUI();
        }

        private float GetLobbyRemaining()
        {
            if (!lobbyCountdownActive)
                return 0f;
            return IsHost
                ? Mathf.Max(0f, lobbyDeadline - Time.unscaledTime)
                : Mathf.Max(0f, clientLobbyRemaining -
                    (Time.unscaledTime - clientLobbyStateReceivedAt));
        }

        private void RefreshLobbyUI()
        {
            roomPanel?.SetLobbyState(
                IsHost ? leftLobbyReady : rightLobbyReady,
                IsHost ? rightLobbyReady : leftLobbyReady,
                Mathf.CeilToInt(GetLobbyRemaining()),
                lobbyCountdownActive);
        }

        private void ResetLobbyState()
        {
            lobbyCountdownActive = false;
            leftLobbyReady = false;
            rightLobbyReady = false;
            lobbyDeadline = 0f;
            clientLobbyRemaining = 0f;
            clientLobbyStateReceivedAt = 0f;
            nextLobbyStateAt = 0f;
        }

        private void SendClientBattleReady()
        {
            if (!IsClient || localBattleReadySent ||
                !networkManager.IsConnectedClient || !messagesRegistered)
                return;

            // Sent only after the client's control choice is locked and its
            // input providers/physics view are ready for host snapshots.
            using var writer = new FastBufferWriter(2, Allocator.Temp);
            writer.WriteValueSafe((byte)1);
            writer.WriteValueSafe((byte)localInputType);
            networkManager.CustomMessagingManager.SendNamedMessage(
                BattleReadyMessage,
                NetworkManager.ServerClientId,
                writer,
                NetworkDelivery.ReliableSequenced);
            localBattleReadySent = true;
            Logger.Info("[Online] Client is ready for the battle countdown.");
        }

        private void OnBattleReadyMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || senderClientId == networkManager.LocalClientId)
                return;

            reader.ReadValueSafe(out byte ready);
            reader.ReadValueSafe(out byte selectedMode);
            if (ready != 1)
                return;

            rightInputType = ToSupportedOnlineInput(selectedMode);
            if (boundBattleManager != null)
                boundBattleManager.RightInputType = rightInputType;
            clientBattleReady = true;
            SendMatchInfoToClients();
            Logger.Info("[Online] Both scene peers can now prepare for battle.");
            TryBeginBattlePreparation();
        }

        private void SendMatchInfo(ulong clientId)
        {
            if (!IsHost || !networkManager.IsListening)
                return;

            using var writer = new FastBufferWriter(1536, Allocator.Temp);
            writer.WriteValueSafe(new FixedString64Bytes(leftDisplayName));
            writer.WriteValueSafe(new FixedString64Bytes(rightDisplayName));
            writer.WriteValueSafe(new FixedString128Bytes(leftAccountId));
            writer.WriteValueSafe(new FixedString128Bytes(rightAccountId));
            writer.WriteValueSafe(new FixedString512Bytes(leftEquipment));
            writer.WriteValueSafe(new FixedString512Bytes(rightEquipment));
            writer.WriteValueSafe((byte)leftInputType);
            writer.WriteValueSafe((byte)rightInputType);
            networkManager.CustomMessagingManager.SendNamedMessage(
                MatchInfoMessage,
                clientId,
                writer,
                NetworkDelivery.ReliableSequenced);
            ApplyOnlineProfiles();
            ApplyOnlineNames();
        }

        private void SendMatchInfoToClients()
        {
            if (!IsHost || networkManager == null || !networkManager.IsListening)
                return;

            foreach (ulong clientId in networkManager.ConnectedClientsIds)
            {
                if (clientId != networkManager.LocalClientId)
                    SendMatchInfo(clientId);
            }
        }

        private void OnMatchInfoMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsClient || senderClientId != NetworkManager.ServerClientId)
                return;

            reader.ReadValueSafe(out FixedString64Bytes leftName);
            reader.ReadValueSafe(out FixedString64Bytes rightName);
            reader.ReadValueSafe(out FixedString128Bytes leftAccount);
            reader.ReadValueSafe(out FixedString128Bytes rightAccount);
            reader.ReadValueSafe(out FixedString512Bytes leftLoadout);
            reader.ReadValueSafe(out FixedString512Bytes rightLoadout);
            reader.ReadValueSafe(out byte leftInputMode);
            reader.ReadValueSafe(out byte rightInputMode);
            leftDisplayName = leftName.IsEmpty ? "Player 1" : leftName.ToString();
            rightDisplayName = rightName.IsEmpty ? "Player 2" : rightName.ToString();
            leftAccountId = leftAccount.IsEmpty ? "player:left" : leftAccount.ToString();
            rightAccountId = rightAccount.IsEmpty ? "player:right" : rightAccount.ToString();
            leftEquipment = leftLoadout.ToString();
            rightEquipment = rightLoadout.ToString();
            leftInputType = ToSupportedOnlineInput(leftInputMode);
            rightInputType = ToSupportedOnlineInput(rightInputMode);
            ApplyOnlineProfiles();
            ApplyOnlineNames();
            if (inputSelectionOpen)
            {
                BattleUIManager.Instance?.UpdateOnlineInputSelection(
                    leftInputType,
                    rightInputType,
                    LocalSide,
                    -1);
            }
        }

        private void OnInputSelectionMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || !inputSelectionOpen || senderClientId == networkManager.LocalClientId)
                return;

            reader.ReadValueSafe(out byte requestedMode);
            rightInputType = ToSupportedOnlineInput(requestedMode);
            if (boundBattleManager != null)
                boundBattleManager.RightInputType = rightInputType;

            SendMatchInfoToClients();
            BattleUIManager.Instance?.UpdateOnlineInputSelection(
                leftInputType,
                rightInputType,
                LocalSide,
                -1);
        }

        private void BeginOnlineBattle()
        {
            if (!IsHost || battleSceneRequested || !clientReady ||
                !lobbyCountdownActive ||
                networkManager == null || networkManager.ConnectedClientsIds.Count < 2 ||
                !(leftLobbyReady && rightLobbyReady || GetLobbyRemaining() <= 0f))
                return;

            Logger.Info(leftLobbyReady && rightLobbyReady
                ? "[Online] Both players ready. Loading Battle."
                : "[Online] Room ready timer expired. Loading Battle.");
            ResetLobbyState();
            ResetInitialStartState();
            // Lobby readiness authorizes the scene transition. Battle starts
            // automatically after both peers load and the 10-second warning.
            startRequested = true;
            battleSceneRequested = true;
            SetStatus("Room ready. Loading battle...");
            roomPanel?.Hide();
            networkManager.SceneManager.LoadScene("Battle", LoadSceneMode.Single);
        }

        private IEnumerator AttachBattleWhenReady()
        {
            yield return null;

            BattleManager battleManager = BattleManager.Instance;
            if (battleManager == null)
            {
                Logger.Error("[Online] BattleManager was not ready after Battle loaded.");
                yield break;
            }

            UnbindBattleManager();
            boundBattleManager = battleManager;
            boundBattleManager.LeftInputType = leftInputType;
            boundBattleManager.RightInputType = rightInputType;
            boundBattleManager.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleStateChanged);
            BindRematchButton();
            BindStartBattleButton();
            ApplyOnlineProfiles();
            ApplyOnlineNames();

            // Both players chose their input mode in the room browser, before
            // entering Battle. Do not add a second 15-second selection phase.
            BattleUIManager.Instance?.LockOnlineInputSelection(leftInputType, rightInputType);

            if (boundBattleManager == null)
                yield break;

            if (IsClient)
            {
                while (boundBattleManager != null && !boundBattleManager.PrepareRemoteClient())
                    yield return null;

                if (boundBattleManager == null)
                    yield break;

                SetClientPhysicsEnabled(false);
                localBattleReady = true;
                SendClientBattleReady();
            }
            else
            {
                hostBattleReady = true;
                TryBeginBattlePreparation();
            }

            RefreshBattleInputVisibility();
            RefreshRematchUI();
            RefreshStartBattleUI();
        }

        private void UnbindBattleManager()
        {
            if (boundBattleManager != null)
                boundBattleManager.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleStateChanged);
            boundBattleManager = null;
            rematchButton = null;
            rematchLabel = null;
            startBattleButton = null;
            startBattleLabel = null;
        }

        private void BindStartBattleButton()
        {
            startBattleButton = Resources.FindObjectsOfTypeAll<Button>()
                .FirstOrDefault(candidate => candidate.name == "BtnStart" &&
                    candidate.gameObject.scene.name == "Battle");
            if (startBattleButton == null)
            {
                Logger.Warning("[Online] BtnStart was not found in Battle.");
                return;
            }

            // The serialized listener calls Battle_Start immediately. Online
            // play instead requires both scene peers and a ten-second warning.
            startBattleButton.onClick = new Button.ButtonClickedEvent();
            startBattleButton.onClick.AddListener(RequestInitialBattleStart);
            startBattleLabel = startBattleButton.GetComponentInChildren<TMP_Text>(true);
            RefreshStartBattleUI();
        }

        public static void RequestStartFromSceneButton()
        {
            if (IsHost)
                Instance.RequestInitialBattleStart();
        }

        private void RequestInitialBattleStart()
        {
            if (!IsHost || boundBattleManager == null ||
                boundBattleManager.CurrentState != BattleState.PreBatle_Preparing ||
                startRequested || preparationActive)
                return;

            startRequested = true;
            Logger.Info("[Online] Host requested battle start; waiting for both scene peers.");
            TryBeginBattlePreparation();
            RefreshStartBattleUI();
        }

        private void TryBeginBattlePreparation()
        {
            if (!IsHost || !startRequested || preparationActive ||
                !hostBattleReady || !clientBattleReady ||
                boundBattleManager == null ||
                boundBattleManager.CurrentState != BattleState.PreBatle_Preparing ||
                networkManager == null || networkManager.ConnectedClientsIds.Count < 2)
                return;

            preparationActive = true;
            preparationDeadline = Time.unscaledTime + BattlePreparationSeconds;
            Logger.Info($"[Online] Both players ready. Battle begins in {BattlePreparationSeconds:0}s.");
            SendBattleSnapshot();
            RefreshStartBattleUI();
        }

        private void StartAuthorizedBattle()
        {
            if (!IsHost || boundBattleManager == null)
                return;

            onlineStartAuthorized = true;
            try
            {
                boundBattleManager.Battle_Start();
            }
            finally
            {
                onlineStartAuthorized = false;
            }
        }

        private float GetPreparationRemaining()
        {
            if (!preparationActive)
                return 0f;

            return IsHost
                ? Mathf.Max(0f, preparationDeadline - Time.unscaledTime)
                : Mathf.Max(0f, clientPreparationRemaining -
                    (Time.unscaledTime - clientPreparationReceivedAt));
        }

        private void RefreshStartBattleUI()
        {
            if (startBattleButton == null || boundBattleManager == null ||
                boundBattleManager.CurrentState != BattleState.PreBatle_Preparing)
                return;

            startBattleButton.interactable = false;
            if (preparationActive)
                BattleUIManager.Instance?.SetOnlinePreparationCountdown(
                    Mathf.CeilToInt(GetPreparationRemaining()));
            if (startBattleLabel == null)
                return;

            if (preparationActive)
                startBattleLabel.SetText($"Starting in {Mathf.CeilToInt(GetPreparationRemaining())}s");
            else
                startBattleLabel.SetText("Preparing controllers...");
        }

        private void ResetInitialStartState()
        {
            hostBattleReady = false;
            clientBattleReady = false;
            localBattleReady = false;
            localBattleReadySent = false;
            startRequested = false;
            preparationActive = false;
            onlineStartAuthorized = false;
            preparationDeadline = 0f;
            clientPreparationRemaining = 0f;
            clientPreparationReceivedAt = 0f;
        }

        private void OnBattleStateChanged(EventParameter _)
        {
            if (boundBattleManager != null)
            {
                if (boundBattleManager.CurrentState == BattleState.PostBattle_ShowResult && IsHost)
                {
                    rematchWindowActive = true;
                    leftRematchRequested = false;
                    rightRematchRequested = false;
                    rematchDeadline = Time.unscaledTime + RematchWindowSeconds;
                }
                else if (boundBattleManager.CurrentState == BattleState.Battle_Preparing)
                {
                    ResetRematchState();
                }
            }

            RefreshBattleInputVisibility();
            ApplyOnlineNames();
            RefreshRematchUI();
        }

        private void BindRematchButton()
        {
            rematchButton = Resources.FindObjectsOfTypeAll<Button>()
                .FirstOrDefault(candidate =>
                    candidate.name == "BtnRematch" && candidate.gameObject.scene.IsValid());

            if (rematchButton == null)
            {
                Logger.Warning("[Online] BtnRematch was not found in Battle.");
                return;
            }

            // Replace the serialized local Battle_Start listener. A client may
            // only request a rematch; only the host can authorize its start.
            rematchButton.onClick = new Button.ButtonClickedEvent();
            rematchButton.onClick.AddListener(OnRematchPressed);
            rematchLabel = rematchButton.GetComponentInChildren<TMP_Text>(true);
        }

        private void OnRematchPressed()
        {
            if (!onlineMatchActive || boundBattleManager == null ||
                boundBattleManager.CurrentState != BattleState.PostBattle_ShowResult ||
                !rematchWindowActive || GetRematchRemaining() <= 0f)
            {
                return;
            }

            if (IsHost)
            {
                leftRematchRequested = true;
                TryStartRematch();
            }
            else
            {
                if (rightRematchRequested || !networkManager.IsConnectedClient)
                    return;

                rightRematchRequested = true;
                using var writer = new FastBufferWriter(sizeof(byte), Allocator.Temp);
                writer.WriteValueSafe((byte)1);
                networkManager.CustomMessagingManager.SendNamedMessage(
                    RematchRequestMessage,
                    NetworkManager.ServerClientId,
                    writer,
                    NetworkDelivery.ReliableSequenced);
            }

            RefreshRematchUI();
        }

        private void OnRematchRequestMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || senderClientId == networkManager.LocalClientId ||
                boundBattleManager == null ||
                boundBattleManager.CurrentState != BattleState.PostBattle_ShowResult ||
                !rematchWindowActive || GetRematchRemaining() <= 0f)
            {
                return;
            }

            reader.ReadValueSafe(out byte requested);
            if (requested != 1)
                return;

            rightRematchRequested = true;
            TryStartRematch();
            RefreshRematchUI();
        }

        private void TryStartRematch()
        {
            if (!IsHost || !rematchWindowActive ||
                !leftRematchRequested || !rightRematchRequested ||
                boundBattleManager == null)
            {
                return;
            }

            rematchWindowActive = false;
            Logger.Info("[Online] Both players accepted the rematch.");
            StartAuthorizedBattle();
        }

        private void ResetRematchState()
        {
            rematchWindowActive = false;
            leftRematchRequested = false;
            rightRematchRequested = false;
            rematchDeadline = 0f;
            clientRematchRemaining = 0f;
            rematchSnapshotReceivedAt = 0f;
        }

        private float GetRematchRemaining()
        {
            if (!rematchWindowActive)
                return 0f;

            return IsHost
                ? Mathf.Max(0f, rematchDeadline - Time.unscaledTime)
                : Mathf.Max(0f, clientRematchRemaining -
                    (Time.unscaledTime - rematchSnapshotReceivedAt));
        }

        private void RefreshRematchUI()
        {
            if (rematchButton == null)
                return;

            bool showingResult = boundBattleManager != null &&
                boundBattleManager.CurrentState == BattleState.PostBattle_ShowResult;
            float remaining = GetRematchRemaining();
            bool localRequested = IsHost ? leftRematchRequested : rightRematchRequested;
            bool opponentRequested = IsHost ? rightRematchRequested : leftRematchRequested;
            rematchButton.interactable = showingResult && rematchWindowActive &&
                remaining > 0f && !localRequested;

            if (rematchLabel == null)
                return;

            if (!showingResult)
                rematchLabel.SetText("Rematch");
            else if (!rematchWindowActive || remaining <= 0f)
                rematchLabel.SetText("Rematch expired");
            else if (localRequested)
                rematchLabel.SetText($"Waiting for opponent ({Mathf.CeilToInt(remaining)}s)");
            else if (opponentRequested)
                rematchLabel.SetText($"Accept Rematch ({Mathf.CeilToInt(remaining)}s)");
            else
                rematchLabel.SetText($"Rematch ({Mathf.CeilToInt(remaining)}s)");
        }

        private void ApplyOnlineNames()
        {
            BattleUIManager.Instance?.SetOnlinePlayerNames(leftDisplayName, rightDisplayName);
        }

        private void ApplyOnlineProfiles()
        {
            GameManager.Instance.ApplyOnlinePlayers(
                leftAccountId,
                leftDisplayName,
                DeserializeEquipment(leftEquipment),
                rightAccountId,
                rightDisplayName,
                DeserializeEquipment(rightEquipment));
        }

        private void RefreshLocalIdentity()
        {
            PlayerAccount account = GameServices.Auth?.Current;
            localAccountId = !string.IsNullOrWhiteSpace(account?.PlayerId)
                ? account.PlayerId
                : profileName;
            localDisplayName = !string.IsNullOrWhiteSpace(account?.DisplayName) &&
                !string.Equals(account.DisplayName, "Player", StringComparison.OrdinalIgnoreCase)
                    ? account.DisplayName
                    : profileName;
            localEquipment = SerializeEquipment(GameServices.PlayerData?.Current?.EquippedBySlot);
        }

        private static InputType ToSupportedOnlineInput(int value)
        {
            InputType input = (InputType)value;
            return input == InputType.LiveCommand ? InputType.LiveCommand : InputType.UI;
        }

        /// <summary>
        /// Consumes the Battle scene dropdown while an online match is active.
        /// Only the dropdown for this process's assigned side can change, and
        /// changes are accepted only during the pre-battle selection window.
        /// </summary>
        public static bool TrySelectOnlineInput(PlayerSide side, int dropdownValue)
        {
            if (!IsActive || SceneManager.GetActiveScene().name != "Battle")
                return false;

            if (Instance == null || !Instance.inputSelectionOpen || side != LocalSide)
                return true;

            InputType selected = ToSupportedOnlineInput(dropdownValue == 1
                ? (int)InputType.LiveCommand
                : (int)InputType.UI);
            Instance.localInputType = selected;
            if (LocalSide == PlayerSide.Left)
                Instance.leftInputType = selected;
            else
                Instance.rightInputType = selected;

            if (Instance.boundBattleManager != null)
            {
                Instance.boundBattleManager.LeftInputType = Instance.leftInputType;
                Instance.boundBattleManager.RightInputType = Instance.rightInputType;
            }

            BattleUIManager.Instance?.UpdateOnlineInputSelection(
                Instance.leftInputType,
                Instance.rightInputType,
                LocalSide,
                -1);

            if (IsHost)
            {
                Instance.SendMatchInfoToClients();
            }
            else if (Instance.networkManager != null &&
                Instance.networkManager.IsConnectedClient &&
                Instance.messagesRegistered)
            {
                using var writer = new FastBufferWriter(sizeof(byte), Allocator.Temp);
                writer.WriteValueSafe((byte)selected);
                Instance.networkManager.CustomMessagingManager.SendNamedMessage(
                    InputSelectionMessage,
                    NetworkManager.ServerClientId,
                    writer,
                    NetworkDelivery.ReliableSequenced);
            }

            return true;
        }

        private static string SerializeEquipment(IReadOnlyDictionary<string, string> equipment)
        {
            if (equipment == null || equipment.Count == 0)
                return string.Empty;

            return string.Join("|", equipment
                .Where(pair => !string.IsNullOrEmpty(pair.Key) && !string.IsNullOrEmpty(pair.Value))
                .OrderBy(pair => pair.Key)
                .Select(pair => $"{pair.Key}={pair.Value}"));
        }

        private static Dictionary<string, string> DeserializeEquipment(string serialized)
        {
            var equipment = new Dictionary<string, string>();
            if (string.IsNullOrEmpty(serialized))
                return equipment;

            foreach (string entry in serialized.Split('|'))
            {
                int separator = entry.IndexOf('=');
                if (separator <= 0 || separator >= entry.Length - 1)
                    continue;

                equipment[entry.Substring(0, separator)] = entry.Substring(separator + 1);
            }

            return equipment;
        }

        public static void RefreshBattleInputVisibility()
        {
            if (!IsActive || InputManager.Instance == null)
                return;

            bool canControl = BattleManager.Instance != null &&
                BattleManager.Instance.CurrentState >= BattleState.Battle_Preparing &&
                BattleManager.Instance.CurrentState < BattleState.PostBattle_ShowResult;

            InputType localMode = LocalSide == PlayerSide.Left
                ? Instance.leftInputType
                : Instance.rightInputType;
            bool buttons = localMode != InputType.LiveCommand;
            bool liveCommands = localMode == InputType.LiveCommand;

            if (InputManager.Instance.LeftButton != null)
                InputManager.Instance.LeftButton.SetActive(
                    canControl && LocalSide == PlayerSide.Left && buttons);
            if (InputManager.Instance.RightButton != null)
                InputManager.Instance.RightButton.SetActive(
                    canControl && LocalSide == PlayerSide.Right && buttons);
            if (InputManager.Instance.LeftLiveCommand != null)
                InputManager.Instance.LeftLiveCommand.SetActive(
                    canControl && LocalSide == PlayerSide.Left && liveCommands);
            if (InputManager.Instance.RightLiveCommand != null)
                InputManager.Instance.RightLiveCommand.SetActive(
                    canControl && LocalSide == PlayerSide.Right && liveCommands);
        }

        public static bool TryRouteLocalInput(InputProvider provider, ISumoAction action)
        {
            if (!IsActive || provider == null || action == null)
                return false;

            // Online players may only issue commands for their assigned side.
            if (provider.PlayerSide != LocalSide)
                return true;

            // The host is authoritative and can enqueue its own left-side input locally.
            if (IsHost)
                return false;

            Instance.SendInput(action);
            return true;
        }

        private void SendInput(ISumoAction action)
        {
            if (!IsClient || !networkManager.IsConnectedClient ||
                BattleManager.Instance == null ||
                BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing)
            {
                return;
            }

            if (action.Type is ActionType.Accelerate or ActionType.TurnLeft or ActionType.TurnRight)
            {
                if (lastInputSentAt.TryGetValue(action.Type, out float lastSent) &&
                    Time.unscaledTime - lastSent < ContinuousInputInterval)
                {
                    return;
                }

                lastInputSentAt[action.Type] = Time.unscaledTime;
            }

            using var writer = new FastBufferWriter(sizeof(int) + sizeof(float), Allocator.Temp);
            writer.WriteValueSafe((int)action.Type);
            writer.WriteValueSafe(action.Duration);
            networkManager.CustomMessagingManager.SendNamedMessage(
                InputMessage,
                NetworkManager.ServerClientId,
                writer,
                NetworkDelivery.ReliableSequenced);
        }

        private void OnInputMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsHost || senderClientId == networkManager.LocalClientId)
                return;

            reader.ReadValueSafe(out int actionValue);
            reader.ReadValueSafe(out float duration);

            if (!Enum.IsDefined(typeof(ActionType), actionValue) ||
                !float.IsFinite(duration) || duration < 0f || duration > 5f)
            {
                Logger.Warning($"[Online] Rejected invalid input from client {senderClientId}.");
                return;
            }

            BattleManager battleManager = BattleManager.Instance;
            if (battleManager == null || battleManager.CurrentState != BattleState.Battle_Ongoing)
                return;

            InputProvider provider = battleManager.Battle.RightPlayer?.InputProvider;
            if (provider == null)
                return;

            ISumoAction action = CreateAction((ActionType)actionValue, duration);
            if (action != null)
                provider.EnqueueNetworkCommand(action);
        }

        private static ISumoAction CreateAction(ActionType type, float duration)
        {
            return type switch
            {
                ActionType.Accelerate => new AccelerateAction(InputType.UI, duration),
                ActionType.TurnLeft => new TurnAction(InputType.UI, ActionType.TurnLeft, duration),
                ActionType.TurnRight => new TurnAction(InputType.UI, ActionType.TurnRight, duration),
                ActionType.Dash => new DashAction(InputType.UI),
                ActionType.SkillBoost => new SkillAction(InputType.UI, ActionType.SkillBoost),
                ActionType.SkillStone => new SkillAction(InputType.UI, ActionType.SkillStone),
                _ => null
            };
        }

        public static void BroadcastDashVfx(PlayerSide side)
        {
            if (IsHost)
                Instance.SendVfx(DashVfx, side, Vector2.zero, 0f);
        }

        public static void BroadcastCollisionVfx(Vector2 position, float speed)
        {
            if (IsHost)
                Instance.SendVfx(CollisionVfx, PlayerSide.Left, position, speed);
        }

        private void SendVfx(byte kind, PlayerSide side, Vector2 position, float speed)
        {
            if (networkManager == null || !networkManager.IsListening ||
                networkManager.ConnectedClientsIds.Count < 2)
                return;

            using var writer = new FastBufferWriter(32, Allocator.Temp);
            writer.WriteValueSafe(kind);
            writer.WriteValueSafe((byte)side);
            writer.WriteValueSafe(position.x);
            writer.WriteValueSafe(position.y);
            writer.WriteValueSafe(speed);
            foreach (ulong clientId in networkManager.ConnectedClientsIds)
            {
                if (clientId != networkManager.LocalClientId)
                    networkManager.CustomMessagingManager.SendNamedMessage(
                        VfxMessage, clientId, writer, NetworkDelivery.ReliableSequenced);
            }
        }

        private void OnVfxMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsClient || senderClientId != NetworkManager.ServerClientId ||
                VFXManager.Instance == null)
                return;

            reader.ReadValueSafe(out byte kind);
            reader.ReadValueSafe(out byte sideValue);
            reader.ReadValueSafe(out float x);
            reader.ReadValueSafe(out float y);
            reader.ReadValueSafe(out float speed);

            if (kind == CollisionVfx)
            {
                VFXManager.Instance.PlayCollisionSpark(new Vector2(x, y), speed);
            }
            else if (kind == DashVfx && sideValue <= (byte)PlayerSide.Right)
            {
                BattleManager battleManager = BattleManager.Instance;
                SumoController controller = sideValue == (byte)PlayerSide.Left
                    ? battleManager?.Battle?.LeftPlayer : battleManager?.Battle?.RightPlayer;
                if (controller != null)
                    VFXManager.Instance.PlayDash(controller.transform, controller.transform.up);
            }
        }

        private void SendBattleSnapshot()
        {
            if (!IsHost || !networkManager.IsListening || networkManager.ConnectedClientsIds.Count < 2)
                return;

            BattleManager battleManager = BattleManager.Instance;
            if (battleManager?.Battle?.LeftPlayer == null || battleManager.Battle.RightPlayer == null)
                return;

            SumoController left = battleManager.Battle.LeftPlayer;
            SumoController right = battleManager.Battle.RightPlayer;
            int roundNumber = battleManager.Battle.CurrentRound?.RoundNumber ?? 0;
            BattleWinner roundWinner = roundNumber > 0
                ? battleManager.Battle.GetRoundWinner(roundNumber)
                : BattleWinner.Draw;

            using var writer = new FastBufferWriter(256, Allocator.Temp);
            writer.WriteValueSafe((int)battleManager.CurrentState);
            writer.WriteValueSafe(battleManager.ElapsedTime);
            writer.WriteValueSafe(battleManager.CountdownRemaining);
            writer.WriteValueSafe(roundNumber);
            writer.WriteValueSafe(battleManager.Battle.LeftWinCount);
            writer.WriteValueSafe(battleManager.Battle.RightWinCount);
            writer.WriteValueSafe((int)roundWinner);
            writer.WriteValueSafe((byte)(rematchWindowActive ? 1 : 0));
            writer.WriteValueSafe((byte)(leftRematchRequested ? 1 : 0));
            writer.WriteValueSafe((byte)(rightRematchRequested ? 1 : 0));
            writer.WriteValueSafe(GetRematchRemaining());
            writer.WriteValueSafe((byte)(preparationActive ? 1 : 0));
            writer.WriteValueSafe(GetPreparationRemaining());
            WriteControllerSnapshot(writer, left);
            WriteControllerSnapshot(writer, right);

            foreach (ulong clientId in networkManager.ConnectedClientsIds)
            {
                if (clientId == networkManager.LocalClientId)
                    continue;

                networkManager.CustomMessagingManager.SendNamedMessage(
                    SnapshotMessage,
                    clientId,
                    writer,
                    NetworkDelivery.UnreliableSequenced);
            }
        }

        private static void WriteControllerSnapshot(FastBufferWriter writer, SumoController controller)
        {
            Vector3 position = controller.transform.position;
            Rigidbody2D body = controller.RigidBody;
            writer.WriteValueSafe(position.x);
            writer.WriteValueSafe(position.y);
            writer.WriteValueSafe(controller.transform.eulerAngles.z);
            writer.WriteValueSafe(body != null ? body.linearVelocity.x : 0f);
            writer.WriteValueSafe(body != null ? body.linearVelocity.y : 0f);
            writer.WriteValueSafe(body != null ? body.angularVelocity : 0f);
            byte visualFlags = 0;
            if (controller.IsAcceleratingVisual) visualFlags |= AcceleratingFlag;
            if (controller.TurnVisualDirection > 0) visualFlags |= TurnLeftFlag;
            if (controller.TurnVisualDirection < 0) visualFlags |= TurnRightFlag;
            if (controller.IsDashActive) visualFlags |= DashActiveFlag;
            if (controller.Skill != null && controller.Skill.IsActive) visualFlags |= SkillActiveFlag;
            writer.WriteValueSafe(visualFlags);
            writer.WriteValueSafe((byte)(controller.Skill?.Type ?? SkillType.Boost));
            writer.WriteValueSafe(controller.Skill != null ?
                Mathf.Clamp01(controller.Skill.CooldownNormalized) : 0f);
            writer.WriteValueSafe(Mathf.Clamp01(controller.DashCooldownNormalized));
        }

        private void OnSnapshotMessage(ulong senderClientId, FastBufferReader reader)
        {
            if (!IsClient || senderClientId != NetworkManager.ServerClientId)
                return;

            reader.ReadValueSafe(out int stateValue);
            reader.ReadValueSafe(out float elapsed);
            reader.ReadValueSafe(out float countdownRemaining);
            reader.ReadValueSafe(out int roundNumber);
            reader.ReadValueSafe(out int leftWins);
            reader.ReadValueSafe(out int rightWins);
            reader.ReadValueSafe(out int winnerValue);
            reader.ReadValueSafe(out byte rematchActive);
            reader.ReadValueSafe(out byte leftRequested);
            reader.ReadValueSafe(out byte rightRequested);
            reader.ReadValueSafe(out float rematchRemaining);
            reader.ReadValueSafe(out byte preparing);
            reader.ReadValueSafe(out float preparationRemaining);
            rematchWindowActive = rematchActive == 1;
            leftRematchRequested = leftRequested == 1;
            rightRematchRequested = rightRequested == 1;
            clientRematchRemaining = Mathf.Max(0f, rematchRemaining);
            rematchSnapshotReceivedAt = Time.unscaledTime;
            preparationActive = preparing == 1;
            clientPreparationRemaining = Mathf.Max(0f, preparationRemaining);
            clientPreparationReceivedAt = Time.unscaledTime;
            leftTarget = ReadControllerSnapshot(reader);
            rightTarget = ReadControllerSnapshot(reader);

            BattleManager battleManager = BattleManager.Instance;
            if (battleManager?.Battle?.LeftPlayer != null && battleManager.Battle.RightPlayer != null)
            {
                ApplyControllerPresentation(battleManager.Battle.LeftPlayer, leftTarget);
                ApplyControllerPresentation(battleManager.Battle.RightPlayer, rightTarget);
            }

            if (!Enum.IsDefined(typeof(BattleState), stateValue) ||
                !Enum.IsDefined(typeof(BattleWinner), winnerValue))
            {
                return;
            }

            // Do not let an early pre-battle snapshot initialize the client's
            // InputProviders before both players' 15-second choices are final.
            if (inputSelectionOpen)
                return;

            boundBattleManager?.ApplyRemoteSnapshot(
                (BattleState)stateValue,
                elapsed,
                countdownRemaining,
                roundNumber,
                leftWins,
                rightWins,
                (BattleWinner)winnerValue);
            RefreshRematchUI();
        }

        private static SnapshotTarget ReadControllerSnapshot(FastBufferReader reader)
        {
            reader.ReadValueSafe(out float x);
            reader.ReadValueSafe(out float y);
            reader.ReadValueSafe(out float rotation);
            reader.ReadValueSafe(out float velocityX);
            reader.ReadValueSafe(out float velocityY);
            reader.ReadValueSafe(out float angularVelocity);
            reader.ReadValueSafe(out byte visualFlags);
            reader.ReadValueSafe(out byte skillType);
            reader.ReadValueSafe(out float skillCooldown);
            reader.ReadValueSafe(out float dashCooldown);
            return new SnapshotTarget(
                new Vector2(x, y),
                rotation,
                new Vector2(velocityX, velocityY),
                angularVelocity,
                visualFlags,
                skillType,
                skillCooldown,
                dashCooldown,
                true);
        }

        private static void ApplyControllerPresentation(SumoController controller, SnapshotTarget target)
        {
            if (!target.Valid || controller == null)
                return;

            SkillType skillType = Enum.IsDefined(typeof(SkillType), (int)target.SkillType)
                ? (SkillType)target.SkillType : SkillType.Boost;
            controller.ApplyRemotePresentation(skillType, target.SkillCooldown,
                target.DashCooldown,
                (target.VisualFlags & SkillActiveFlag) != 0,
                (target.VisualFlags & DashActiveFlag) != 0);
        }

        private void InterpolateClientView()
        {
            BattleManager battleManager = BattleManager.Instance;
            if (battleManager?.Battle?.LeftPlayer == null || battleManager.Battle.RightPlayer == null)
                return;

            ApplyControllerTarget(battleManager.Battle.LeftPlayer, leftTarget);
            ApplyControllerTarget(battleManager.Battle.RightPlayer, rightTarget);
            if (battleManager.CurrentState == BattleState.Battle_Ongoing && VFXManager.Instance != null)
            {
                PlayContinuousVfx(battleManager.Battle.LeftPlayer, leftTarget);
                PlayContinuousVfx(battleManager.Battle.RightPlayer, rightTarget);
            }
        }

        private static void PlayContinuousVfx(SumoController controller, SnapshotTarget target)
        {
            if (!target.Valid || controller == null)
                return;

            Vector2 facing = controller.transform.up;
            if ((target.VisualFlags & AcceleratingFlag) != 0)
                VFXManager.Instance.PlayAccelerationTrail(controller.transform, facing);
            if ((target.VisualFlags & (TurnLeftFlag | TurnRightFlag)) != 0)
                VFXManager.Instance.PlayTurnTrail(controller.transform, facing,
                    (target.VisualFlags & TurnLeftFlag) != 0 ? 1 : -1);
        }

        private static void ApplyControllerTarget(SumoController controller, SnapshotTarget target)
        {
            if (!target.Valid || controller == null)
                return;

            Transform controllerTransform = controller.transform;
            Vector2 current = controllerTransform.position;
            float distance = Vector2.Distance(current, target.Position);
            float t = distance > 1f ? 1f : 1f - Mathf.Exp(-25f * Time.unscaledDeltaTime);
            Vector2 smoothedPosition = Vector2.Lerp(current, target.Position, t);
            float smoothedRotation = Mathf.LerpAngle(controllerTransform.eulerAngles.z, target.Rotation, t);

            controllerTransform.SetPositionAndRotation(
                new Vector3(smoothedPosition.x, smoothedPosition.y, controllerTransform.position.z),
                Quaternion.Euler(0f, 0f, smoothedRotation));

            if (controller.RigidBody != null)
            {
                controller.RigidBody.linearVelocity = target.Velocity;
                controller.RigidBody.angularVelocity = target.AngularVelocity;
            }
        }

        private void SetClientPhysicsEnabled(bool enabled)
        {
            BattleManager battleManager = BattleManager.Instance;
            if (battleManager?.Battle?.LeftPlayer?.RigidBody != null)
                battleManager.Battle.LeftPlayer.RigidBody.simulated = enabled;
            if (battleManager?.Battle?.RightPlayer?.RigidBody != null)
                battleManager.Battle.RightPlayer.RigidBody.simulated = enabled;
        }

        private void OnSessionPlayerJoined(string playerId)
        {
            Logger.Info($"[Online] Player joined: {playerId}");
            if (IsHost)
            {
                SetStatus("Opponent found. Establishing relay...");
                BeginLobbyCountdown();
            }
        }

        private void OnSessionPlayerLeft(string playerId)
        {
            if (!leaving && onlineMatchActive)
                _ = HandleConnectionLostAsync($"Player {playerId} left the match.");
        }

        private void OnRemovedFromSession()
        {
            if (!leaving && onlineMatchActive)
                _ = HandleConnectionLostAsync("You were removed from the session.");
        }

        private void OnSessionDeleted()
        {
            if (!leaving && onlineMatchActive)
                _ = HandleConnectionLostAsync("The online session ended.");
        }

        private void OnClientDisconnected(ulong clientId)
        {
            if (leaving || !onlineMatchActive)
                return;

            if (IsHost && clientId == networkManager.LocalClientId)
                return;

            _ = HandleConnectionLostAsync("The opponent disconnected.");
        }

        private async Task HandleConnectionLostAsync(string reason)
        {
            if (connectionLossHandled || leaving || !onlineMatchActive)
                return;

            connectionLossHandled = true;
            Logger.Warning($"[Online] {reason}");
            bool battleSceneLoaded = SceneManager.GetActiveScene().name == "Battle";
            PlayerSide localWinner = LocalSide;
            bool forfeitShown = false;
            if (battleSceneLoaded)
            {
                // A disconnect can arrive in the frame where the network scene
                // has loaded but the attachment coroutine has not completed.
                // Give BattleManager a short, frame-based chance to initialize.
                int framesRemaining = 60;
                while (BattleManager.Instance == null && framesRemaining-- > 0)
                    await Task.Yield();

                rematchWindowActive = false;
                RefreshRematchUI();
                BattleManager manager = boundBattleManager ?? BattleManager.Instance;
                forfeitShown = manager != null && manager.FinishOnlineByForfeit(localWinner);
            }

            await LeaveSessionAsync(false);
            if (!forfeitShown)
            {
                if (SceneManager.GetActiveScene().name != "MainMenu")
                    SceneManager.LoadScene("MainMenu");
                else
                {
                    EnsureRoomPanel();
                    roomPanel.Show(localInputType);
                    roomPanel.SetWaiting(false);
                    roomPanel.SetStatus($"Opponent left: {reason}");
                }
            }
        }

        public static void LeaveAndReturnToMainMenu()
        {
            if (Instance == null || !IsActive)
            {
                SceneManager.LoadScene("MainMenu");
                return;
            }

            _ = Instance.LeaveSessionAndReturnAsync();
        }

        private async Task LeaveSessionAndReturnAsync()
        {
            await LeaveSessionAsync(true);
            SceneManager.LoadScene("MainMenu");
        }

        private async Task LeaveSessionAsync(bool userRequested)
        {
            if (leaving)
                return;

            leaving = true;

            try
            {
                UnbindBattleManager();
                UnregisterMessages();
                UnsubscribeSessionEvents();

                if (session != null && session.IsMember)
                    await session.LeaveAsync();
            }
            catch (Exception ex)
            {
                Logger.Warning($"[Online] Failed to leave cleanly: {ex.Message}");
            }
            finally
            {
                await EnsureNetworkManagerStoppedAsync();
                session = null;
                matchmaking = false;
                onlineMatchActive = false;
                battleSceneRequested = false;
                clientReady = false;
                localReadySent = false;
                lastInputSentAt.Clear();
                leftTarget = default;
                rightTarget = default;
                leftDisplayName = "Player 1";
                rightDisplayName = "Player 2";
                leftAccountId = "player:left";
                rightAccountId = "player:right";
                leftEquipment = string.Empty;
                rightEquipment = string.Empty;
                localDisplayName = profileName;
                localAccountId = profileName;
                localEquipment = string.Empty;
                leftInputType = InputType.UI;
                rightInputType = InputType.UI;
                inputSelectionOpen = false;
                ResetLobbyState();
                ResetInitialStartState();
                ResetRematchState();
                GameManager.Instance.RestoreLocalProfiles();
                leaving = false;

                if (userRequested)
                    Logger.Info("[Online] Player left the online session.");
            }
        }

        private async Task EnsureNetworkManagerStoppedAsync()
        {
            if (networkManager == null)
                return;

            if (networkManager.IsListening && !networkManager.ShutdownInProgress)
                networkManager.Shutdown(true);

            int framesRemaining = 120;
            while ((networkManager.IsListening || networkManager.ShutdownInProgress) &&
                   framesRemaining-- > 0)
            {
                await Task.Yield();
            }

            if (networkManager.IsListening || networkManager.ShutdownInProgress)
                Logger.Warning("[Online] NetworkManager did not finish shutting down before timeout.");
        }

        private void SetStatus(string text)
        {
            if (onlineStatus != null)
                onlineStatus.text = text;
        }

        private static string FriendlyError(Exception exception)
        {
            string message = exception.Message ?? "Unknown error";
            if (message.Length > 90)
                message = message.Substring(0, 87) + "...";
            return message;
        }

        private readonly struct SnapshotTarget
        {
            public readonly Vector2 Position;
            public readonly float Rotation;
            public readonly Vector2 Velocity;
            public readonly float AngularVelocity;
            public readonly byte VisualFlags;
            public readonly byte SkillType;
            public readonly float SkillCooldown;
            public readonly float DashCooldown;
            public readonly bool Valid;

            public SnapshotTarget(
                Vector2 position,
                float rotation,
                Vector2 velocity,
                float angularVelocity,
                byte visualFlags,
                byte skillType,
                float skillCooldown,
                float dashCooldown,
                bool valid)
            {
                Position = position;
                Rotation = rotation;
                Velocity = velocity;
                AngularVelocity = angularVelocity;
                VisualFlags = visualFlags;
                SkillType = skillType;
                SkillCooldown = skillCooldown;
                DashCooldown = dashCooldown;
                Valid = valid;
            }
        }
    }

    /// <summary>Parses per-instance settings before UGS initialization.</summary>
    internal static class OnlineLaunchOptions
    {
        private const string ProfileArgument = "-ugs-profile";
        private const string EnvironmentArgument = "-ugs-environment";
        private const string MatchPoolArgument = "-match-pool";
        private const string AutoOnlineArgument = "-auto-online";
        private const string AutoCreateRoomArgument = "-auto-create-room";
        private const string AutoJoinRoomArgument = "-auto-join-room";
        private static string cachedProfileName;
        private static FileStream instanceProfileLock;

        public static string GetProfileName()
        {
            if (!string.IsNullOrEmpty(cachedProfileName))
                return cachedProfileName;

            string requested = GetArgument(ProfileArgument);
            if (string.IsNullOrWhiteSpace(requested))
                requested = Application.isEditor ? "sumobot_editor" : AcquireLocalInstanceProfile();

            char[] sanitized = requested
                .Where(character => char.IsLetterOrDigit(character) || character is '-' or '_')
                .Take(30)
                .ToArray();
            string profile = new(sanitized);
            cachedProfileName = string.IsNullOrEmpty(profile) ? "sumobot_local_1" : profile;
            return cachedProfileName;
        }

        private static string AcquireLocalInstanceProfile()
        {
            try
            {
                string directory = Path.Combine(Application.persistentDataPath, "InstanceProfiles");
                Directory.CreateDirectory(directory);
                for (int slot = 1; slot <= 8; slot++)
                {
                    string path = Path.Combine(directory, $"slot-{slot}.lock");
                    try
                    {
                        instanceProfileLock = new FileStream(
                            path,
                            FileMode.OpenOrCreate,
                            FileAccess.ReadWrite,
                            FileShare.None);
                        return $"sumobot_local_{slot}";
                    }
                    catch (IOException)
                    {
                        // Another live game process owns this stable profile slot.
                    }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                Logger.Warning($"[Online] Could not reserve a persistent instance profile: {ex.Message}");
            }

            // Extremely unlikely fallback when all stable slots are occupied.
            return $"sumobot_process_{System.Diagnostics.Process.GetCurrentProcess().Id}";
        }

        public static string GetEnvironmentName() => GetArgument(EnvironmentArgument);

        public static bool HasAutoOnlineFlag()
        {
            return Environment.GetCommandLineArgs().Any(argument =>
                string.Equals(argument, AutoOnlineArgument, StringComparison.OrdinalIgnoreCase));
        }

        public static bool HasAutoCreateRoomFlag()
        {
            return Environment.GetCommandLineArgs().Any(argument =>
                string.Equals(argument, AutoCreateRoomArgument, StringComparison.OrdinalIgnoreCase));
        }

        public static bool HasAutoJoinRoomFlag()
        {
            return Environment.GetCommandLineArgs().Any(argument =>
                string.Equals(argument, AutoJoinRoomArgument, StringComparison.OrdinalIgnoreCase));
        }

        public static string GetMatchPoolName()
        {
            string requested = GetArgument(MatchPoolArgument);
            if (string.IsNullOrWhiteSpace(requested))
                return "sumobot-online-v1";

            char[] sanitized = requested
                .Where(character => char.IsLetterOrDigit(character) || character is '-' or '_')
                .Take(64)
                .ToArray();
            string pool = new(sanitized);
            return string.IsNullOrEmpty(pool) ? "sumobot-online-v1" : pool;
        }

        private static string GetArgument(string name)
        {
            string[] arguments = Environment.GetCommandLineArgs();
            for (int index = 0; index < arguments.Length; index++)
            {
                string argument = arguments[index];
                if (argument.StartsWith(name + "=", StringComparison.OrdinalIgnoreCase))
                    return argument.Substring(name.Length + 1).Trim();

                if (string.Equals(argument, name, StringComparison.OrdinalIgnoreCase) &&
                    index + 1 < arguments.Length)
                {
                    return arguments[index + 1].Trim();
                }
            }

            return string.Empty;
        }
    }
}
