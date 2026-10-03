using System;
using System.Collections;
using System.Collections.Generic;
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
        private const string MatchType = "sumobot-online-v1";
        private const string MatchModeProperty = "mode";
        private const string DisplayNameProperty = "displayName";
        private const string ReadyMessage = "sumobot/ready/v1";
        private const string MatchInfoMessage = "sumobot/match-info/v1";
        private const string InputMessage = "sumobot/input/v1";
        private const string SnapshotMessage = "sumobot/snapshot/v1";
        private const string RematchRequestMessage = "sumobot/rematch/v1";
        private const float SnapshotInterval = 1f / 20f;
        private const float ContinuousInputInterval = 0.075f;
        private const float RematchWindowSeconds = 15f;

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
        private bool matchmaking;
        private bool onlineMatchActive;
        private bool messagesRegistered;
        private bool localReadySent;
        private bool battleSceneRequested;
        private bool leaving;
        private bool clientReady;
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
            Application.runInBackground = true;

            transport = gameObject.AddComponent<UnityTransport>();
            networkManager = gameObject.AddComponent<NetworkManager>();
            // NetworkConfig is serialized when NetworkManager lives in a scene,
            // but NGO leaves it null when the component is created at runtime.
            networkManager.NetworkConfig = new NetworkConfig();
            networkManager.NetworkConfig.NetworkTransport = transport;
            networkManager.NetworkConfig.EnableSceneManagement = true;
            networkManager.NetworkConfig.ProtocolVersion = 1;

            SceneManager.sceneLoaded += OnSceneLoaded;
            networkManager.OnClientConnectedCallback += OnClientConnected;
            networkManager.OnClientDisconnectCallback += OnClientDisconnected;
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

            if (!onlineMatchActive || SceneManager.GetActiveScene().name != "Battle")
                return;

            if (IsHost)
            {
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
                InterpolateClientView();
            }

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
                StartCoroutine(AttachBattleWhenReady());
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
                roomPanel?.SetStatus("Connected. Preparing battle...");
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

            for (int attempt = 0; attempt < 20 && !onlineMatchActive; attempt++)
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
                    return;
                }

                await Task.Delay(500);
            }

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
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(MatchInfoMessage, OnMatchInfoMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(InputMessage, OnInputMessage);
            networkManager.CustomMessagingManager.RegisterNamedMessageHandler(SnapshotMessage, OnSnapshotMessage);
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
                SetStatus("Connected. Preparing battle...");
                SendClientReady();
            }
        }

        private void UnregisterMessages()
        {
            if (!messagesRegistered || networkManager == null || networkManager.CustomMessagingManager == null)
                return;

            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(ReadyMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(MatchInfoMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(InputMessage);
            networkManager.CustomMessagingManager.UnregisterNamedMessageHandler(SnapshotMessage);
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
            if (clientReady && session != null && session.PlayerCount >= 2)
                BeginOnlineBattle();
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
        }

        private void BeginOnlineBattle()
        {
            if (!IsHost || battleSceneRequested || !clientReady)
                return;

            battleSceneRequested = true;
            SetStatus("Opponent found. Loading battle...");
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
            BattleUIManager.Instance?.ApplyOnlineInputSelection(leftInputType, rightInputType);
            boundBattleManager.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleStateChanged);
            BindRematchButton();
            ApplyOnlineProfiles();
            ApplyOnlineNames();

            if (IsClient)
            {
                while (boundBattleManager != null && !boundBattleManager.PrepareRemoteClient())
                    yield return null;

                if (boundBattleManager == null)
                    yield break;

                SetClientPhysicsEnabled(false);
            }
            else
            {
                yield return null;
                boundBattleManager.Battle_Start();
            }

            RefreshBattleInputVisibility();
            RefreshRematchUI();
        }

        private void UnbindBattleManager()
        {
            if (boundBattleManager != null)
                boundBattleManager.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleStateChanged);
            boundBattleManager = null;
            rematchButton = null;
            rematchLabel = null;
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
            boundBattleManager.Battle_Start();
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
            rematchWindowActive = rematchActive == 1;
            leftRematchRequested = leftRequested == 1;
            rightRematchRequested = rightRequested == 1;
            clientRematchRemaining = Mathf.Max(0f, rematchRemaining);
            rematchSnapshotReceivedAt = Time.unscaledTime;
            leftTarget = ReadControllerSnapshot(reader);
            rightTarget = ReadControllerSnapshot(reader);

            if (!Enum.IsDefined(typeof(BattleState), stateValue) ||
                !Enum.IsDefined(typeof(BattleWinner), winnerValue))
            {
                return;
            }

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
            return new SnapshotTarget(
                new Vector2(x, y),
                rotation,
                new Vector2(velocityX, velocityY),
                angularVelocity,
                true);
        }

        private void InterpolateClientView()
        {
            BattleManager battleManager = BattleManager.Instance;
            if (battleManager?.Battle?.LeftPlayer == null || battleManager.Battle.RightPlayer == null)
                return;

            ApplyControllerTarget(battleManager.Battle.LeftPlayer, leftTarget);
            ApplyControllerTarget(battleManager.Battle.RightPlayer, rightTarget);
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
                SetStatus("Opponent found. Establishing relay...");
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
            public readonly bool Valid;

            public SnapshotTarget(
                Vector2 position,
                float rotation,
                Vector2 velocity,
                float angularVelocity,
                bool valid)
            {
                Position = position;
                Rotation = rotation;
                Velocity = velocity;
                AngularVelocity = angularVelocity;
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

        public static string GetProfileName()
        {
            string requested = GetArgument(ProfileArgument);
            if (string.IsNullOrWhiteSpace(requested))
                requested = Application.isEditor ? "sumobot_editor" : "sumobot_default";

            char[] sanitized = requested
                .Where(character => char.IsLetterOrDigit(character) || character is '-' or '_')
                .Take(30)
                .ToArray();
            string profile = new(sanitized);
            return string.IsNullOrEmpty(profile) ? "sumobot_default" : profile;
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
