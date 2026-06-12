using System.Collections.Generic;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

public class AIBot_NN_Hasan : Bot
{
    private enum TacticalState
    {
        Recover,
        EdgeAttack,
        FlankDash,
        FrontDash,
        StoneBlock,
        CounterDash,
        TurnLeftAttack,
        TurnRightAttack,
        RotateTrack,
        Search
    }

    private enum EnemyAction
    {
        Rush,
        Circle,
        Retreat,
        EdgeVulnerable,
        SideExposed,
        Waiting
    }

    public override string ID => "NN_Hasan";
    public override SkillType DefaultSkillType => SkillType.Stone;

    public bool loadModel = true;
    public bool saveModel = false;
    public string modelFileName = "NN_Hasan_Model_V3";
    public string fallbackModelFileName = "";
    public float learningRate = 0.02f;
    public float nnTurnThreshold = 0.05f;
    public float nnAccelerateThreshold = 0.0f;
    public float nnDashThreshold = 0.01f;
    public float nnSkillThreshold = 0.05f;
    public int input = 28;
    public int hidden = 32;
    public int output = 5;

    private const float DangerEdge = 0.54f;
    private const float CautionEdge = 0.5f;
    private const float SafeCenter = 0.2f;
    private const float CenterControlEdge = 0.46f;
    private const float OutwardVelocityEdge = 0.44f;
    private const float EnemyEdgeAttack = 0.32f;
    private const float EnemyAlmostOut = 0.48f;
    private const float PredictionTime = 0.1f;
    private const float FaceEnemyAngle = 28f;
    private const float CounterAngle = 60f;
    private const float QuickDashAngle = 38f;
    private const float SideAttackAngle = 82f;
    private const float StoneBlockDistance = 1.45f;
    private const float CounterPushDistance = 4.2f;
    private const float PressureDistance = 5.2f;
    private const float SearchPulseDuration = 0.12f;
    private const float PushDuration = 0.45f;
    private const float PressureDuration = 0.3f;
    private const float RecoveryDuration = 0.1f;
    private const float SafeSearchLanding = 0.34f;
    private const float SafePushLanding = 0.82f;
    private const float SafeDashLanding = 0.82f;
    private const float DashInterval = 0.12f;
    private const float OpeningCenterDuration = 0.36f;
    private const float OpeningMoveDuration = 0.24f;
    private const float OpeningBackDistance = 1.25f;
    private const float OpeningMaxTime = 0.45f;
    private const float MinTurnDuration = 0.1f;
    private const float MaxTurnDuration = 0.16f;
    private const float CounterWindow = 0.35f;
    private const float SlowEnemySpeed = 0.55f;
    private const float FastEnemySpeed = 1.2f;
    private const float ProfileBlend = 0.18f;
    private const float LateGameStartRatio = 0.55f;
    private const float LateGamePushBonus = 0.22f;
    private const float LateGameAngleBonus = 18f;
    private const float LateGameSafetyBonus = 0.08f;
    private const float LateGameCenterHoldEdge = 0.08f;
    private const float LateGameCenterRushDuration = 0.45f;
    private const float LateGameCenterHardTurnAngle = 115f;
    private const float LateGameEnemyBlockDistance = 3.2f;
    private const float NoFlankEdge = 0.32f;
    private const float EdgeAngleLockEdge = 0.34f;
    private const float EdgeSafeFacingDot = 0.62f;
    private const float EdgeLockPushDuration = 0.16f;
    private const float InwardProgressMargin = 0.01f;

    private SumoAPI api;
    private bool returningToCenter;
    private bool openingAttackUsed;
    private float nextDashTime;
    private float counterDashUntil;
    private float openingStartedAt;
    private float enemyRushScore;
    private float enemyCircleScore;
    private float enemyRetreatScore;
    private float enemyEdgeScore;
    private float[] previousInputs;
    private float previousMyEdge;
    private float previousEnemyEdge;
    private float previousDistance;
    private float lastDashTime = -99f;
    private float lastStoneTime = -99f;
    private float lastCollisionTime = -99f;
    private bool lastCollisionWasActor;
    private int searchDirection = 1;
    private NeuralNetwork NN;

    public override void OnBotInit(SumoAPI botAPI)
    {
        api = botAPI;
        returningToCenter = false;
        openingAttackUsed = false;
        nextDashTime = 0f;
        counterDashUntil = 0f;
        openingStartedAt = -1f;
        enemyRushScore = 0f;
        enemyCircleScore = 0f;
        enemyRetreatScore = 0f;
        enemyEdgeScore = 0f;
        previousInputs = null;
        previousMyEdge = 0f;
        previousEnemyEdge = 0f;
        previousDistance = 0f;
        lastDashTime = -99f;
        lastStoneTime = -99f;
        lastCollisionTime = -99f;
        lastCollisionWasActor = false;
        searchDirection = 1;

        InitNeuralNetwork();
    }

    public override void OnBotUpdate()
    {
        ClearCommands();

        if (api.BattleInfo.CurrentState != BattleState.Battle_Ongoing)
            return;

        float myEdge = EdgeRatio(api.MyRobot.Position);
        float angleToEnemy = api.Angle();
        float angleToCenter = api.Angle(targetPos: api.BattleInfo.ArenaPosition);
        float distanceToEnemy = Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
        Vector2 predictedEnemy = PredictEnemyPosition(PredictionTime);
        float predictedEnemyEdge = EdgeRatio(predictedEnemy);

        UpdateEnemyProfile();

        if (!openingAttackUsed)
        {
            openingAttackUsed = OpeningCenterRush(myEdge, angleToCenter);
            Submit();
            return;
        }

        if (ShouldEdgeAngleLock(myEdge))
        {
            returningToCenter = true;
            EdgeAngleLockRecover(angleToCenter);
            Submit();
            return;
        }

        if (LateGameAggression() > 0f && myEdge > LateGameCenterHoldEdge)
        {
            returningToCenter = false;
            LateGameFastResponse(myEdge, angleToCenter, angleToEnemy, distanceToEnemy);
            Submit();
            return;
        }

        if (myEdge > NoFlankEdge && distanceToEnemy <= LateGameEnemyBlockDistance && Mathf.Abs(angleToEnemy) <= DynamicCounterAngle())
        {
            returningToCenter = false;
            LateGameFastResponse(myEdge, angleToCenter, angleToEnemy, distanceToEnemy);
            Submit();
            return;
        }

        if (myEdge > CautionEdge || ShouldCancelForOutwardVelocity(myEdge))
            returningToCenter = true;

        if (returningToCenter)
        {
            RecoverToCenter(myEdge, angleToCenter);
            Submit();
            return;
        }

        ThinkAndActWithNeuralNetwork(myEdge, angleToEnemy, distanceToEnemy, predictedEnemy, predictedEnemyEdge);

        Submit();
    }

    public override void OnBotCollision(BounceEvent bounceEvent)
    {
        ClearCommands();
        lastCollisionTime = CurrentTime();
        lastCollisionWasActor = bounceEvent.MyInfo.IsActor;

        if (!bounceEvent.MyInfo.IsActor)
        {
            returningToCenter = true;
            searchDirection *= -1;
        }
    }

    public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
    {
        if (state == BattleState.Battle_Ongoing)
        {
            returningToCenter = false;
            openingAttackUsed = false;
            nextDashTime = 0f;
            counterDashUntil = 0f;
            openingStartedAt = -1f;
            enemyRushScore = 0f;
            enemyCircleScore = 0f;
            enemyRetreatScore = 0f;
            enemyEdgeScore = 0f;
            previousInputs = null;
            previousMyEdge = 0f;
            previousEnemyEdge = 0f;
            previousDistance = 0f;
            lastDashTime = -99f;
            lastStoneTime = -99f;
            lastCollisionTime = -99f;
            lastCollisionWasActor = false;
        }

#if UNITY_EDITOR
        if (state == BattleState.Battle_End && saveModel && NN != null)
        {
            string path = "Assets/Resources/Bot/Example/Micro-Competition/" + modelFileName + ".json";
			//"Assets/Resources/ML/Models/NN/" + modelFileName + ".json";

			NN.Save(path);
            Logger.Info($"Saved Hasan NN to {path}");
        }
#endif
    }

    private void InitNeuralNetwork()
    {
        string path = "Bot/Example/Micro-Competition/" + modelFileName;
		//"ML/Models/NN/" + modelFileName;

		if (loadModel)
        {
            try
            {
                NN = NeuralNetwork.Load(path);
                Logger.Info($"Loaded Hasan NN from {path}");
                return;
            }
            catch (System.Exception exception)
            {
                Logger.Info($"Could not load Hasan NN from {path}. {exception.Message}");
            }

            if (!string.IsNullOrEmpty(fallbackModelFileName))
            {
                string fallbackPath = "Bot/Example/Micro-Competition/" + fallbackModelFileName;
				//"ML/Models/NN/" + fallbackModelFileName;
				try
                {
                    NN = NeuralNetwork.Load(fallbackPath);
                    Logger.Info($"Loaded Hasan fallback NN from {fallbackPath}");
                    return;
                }
                catch (System.Exception exception)
                {
                    Logger.Info($"Could not load Hasan fallback NN from {fallbackPath}. Creating new network. {exception.Message}");
                }
            }
        }

        NN = new NeuralNetwork(input, hidden, output);
        Logger.Info("Created new Hasan NN V3");
    }

    private void ThinkAndActWithNeuralNetwork(
        float myEdge,
        float angleToEnemy,
        float distanceToEnemy,
        Vector2 predictedEnemy,
        float predictedEnemyEdge)
    {
        if (NN == null)
            InitNeuralNetwork();

        float[] inputs = BuildNeuralInputs(angleToEnemy, distanceToEnemy, predictedEnemyEdge);
        float[] outputs = NN.Forward(inputs);
        float[] targets = BuildTrainingTargets(myEdge, angleToEnemy, distanceToEnemy, predictedEnemyEdge);

        TrainPreviousActionFromOutcome(myEdge, distanceToEnemy, predictedEnemyEdge);
        ApplyNeuralOutputs(outputs, myEdge, angleToEnemy, distanceToEnemy, predictedEnemy, predictedEnemyEdge);
        NN.Train(inputs, targets, learningRate);

        previousInputs = (float[])inputs.Clone();
        previousMyEdge = myEdge;
        previousEnemyEdge = predictedEnemyEdge;
        previousDistance = distanceToEnemy;
    }

    private float[] BuildNeuralInputs(float angleToEnemy, float distanceToEnemy, float predictedEnemyEdge)
    {
        float posX = api.MyRobot.Position.x / api.BattleInfo.ArenaRadius;
        float posY = api.MyRobot.Position.y / api.BattleInfo.ArenaRadius;
        float distanceNormalized = api.DistanceNormalized();
        float angleNormalized = Mathf.Clamp(angleToEnemy / 180f, -1f, 1f);
        float myEdge = EdgeRatio(api.MyRobot.Position);
        float outwardVelocity = OutwardVelocityScore(api.MyRobot.Position, api.MyRobot.LinearVelocity);
        float enemyTowardMe = EnemyTowardMeScore();
        float enemyFacingMe = EnemyFacingMeScore();
        float dashReady = api.MyRobot.IsDashOnCooldown ? 0f : 1f;
        float skillReady = api.MyRobot.Skill.IsSkillOnCooldown ? 0f : 1f;
        float relativeClosingSpeed = RelativeClosingSpeedScore();
        float enemyOutwardVelocity = OutwardVelocityScore(api.EnemyRobot.Position, api.EnemyRobot.LinearVelocity);
        float hasanCenterControl = HasanBetweenEnemyAndCenterScore();
        float enemyCenterControl = EnemyBetweenHasanAndCenterScore();
        float recentCollision = RecentCollisionScore();
        float collisionActor = lastCollisionWasActor ? 1f : 0f;
        float timeSinceDash = TimeSinceScore(lastDashTime, 2f);
        float timeSinceStone = TimeSinceScore(lastStoneTime, 2f);
        float enemySpeedSpike = EnemySpeedSpikeScore();
        float angleToCenter = Mathf.Clamp(api.Angle(targetPos: api.BattleInfo.ArenaPosition) / 180f, -1f, 1f);
        float predictedAccelEdge = PredictedEdgeAfterAccelerate(PressureDuration);
        float predictedDashEdge = PredictedEdgeAfterDash();
        float enemyFacingOutside = EnemyFacingOutsideScore();
        float lateGame = LateGameAggression();

        return new float[]
        {
            posX,
            posY,
            angleNormalized,
            distanceNormalized,
            myEdge,
            predictedEnemyEdge,
            outwardVelocity,
            enemyTowardMe,
            enemyFacingMe,
            enemyRushScore,
            enemyCircleScore,
            enemyRetreatScore,
            dashReady,
            skillReady,
            relativeClosingSpeed,
            enemyOutwardVelocity,
            hasanCenterControl,
            enemyCenterControl,
            recentCollision,
            collisionActor,
            timeSinceDash,
            timeSinceStone,
            enemySpeedSpike,
            angleToCenter,
            predictedAccelEdge,
            predictedDashEdge,
            enemyFacingOutside,
            lateGame
        };
    }

    private float[] BuildTrainingTargets(float myEdge, float angleToEnemy, float distanceToEnemy, float predictedEnemyEdge)
    {
        float[] targets = new float[5];
        float absAngle = Mathf.Abs(angleToEnemy);
        bool safeToPressure = CanPressure(myEdge, distanceToEnemy);
        bool safeToAttack = CanAggressiveAttack(myEdge, distanceToEnemy);
        bool enemyNearEdge = predictedEnemyEdge >= EnemyEdgeAttack || enemyEdgeScore > 0.55f;
        bool enemyMovingOutward = OutwardVelocityScore(api.EnemyRobot.Position, api.EnemyRobot.LinearVelocity) > 0.62f;
        bool badCenterControl = EnemyBetweenHasanAndCenterScore() > 0.65f && myEdge > CenterControlEdge;
        bool shouldStone = ShouldStoneBlock(distanceToEnemy, myEdge);
        bool danger = myEdge > CautionEdge || ShouldCancelForOutwardVelocity(myEdge) || badCenterControl;
        float attackAngle = DynamicCounterAngle();
        float lateGame = LateGameAggression();

        targets[0] = !danger && safeToPressure && absAngle <= attackAngle ? 1f : 0f;
        targets[1] = angleToEnemy > 0f ? Mathf.Clamp01(absAngle / 90f) : 0f;
        targets[2] = angleToEnemy < 0f ? Mathf.Clamp01(absAngle / 90f) : 0f;
        targets[3] = !danger && safeToAttack && absAngle <= attackAngle && (lateGame > 0.3f || enemyNearEdge || enemyMovingOutward || enemyRetreatScore > 0.35f || distanceToEnemy < StoneBlockDistance) ? 1f : 0f;
        targets[4] = shouldStone || badCenterControl ? 1f : 0f;

        return targets;
    }

    private void TrainPreviousActionFromOutcome(float myEdge, float distanceToEnemy, float predictedEnemyEdge)
    {
        if (previousInputs == null)
            return;

        float edgeDelta = myEdge - previousMyEdge;
        float enemyEdgeDelta = predictedEnemyEdge - previousEnemyEdge;
        float distanceDelta = distanceToEnemy - previousDistance;
        bool badOutcome = edgeDelta > 0.04f || (myEdge > CautionEdge && IsMovingOutward());
        bool goodOutcome = enemyEdgeDelta > 0.04f || distanceDelta < -0.25f;

        if (!badOutcome && !goodOutcome)
            return;

        float[] outcomeTargets = new float[5];

        if (badOutcome)
        {
            float centerAngle = api.Angle(targetPos: api.BattleInfo.ArenaPosition);
            outcomeTargets[0] = 0f;
            outcomeTargets[1] = centerAngle > 0f ? Mathf.Clamp01(Mathf.Abs(centerAngle) / 90f) : 0f;
            outcomeTargets[2] = centerAngle < 0f ? Mathf.Clamp01(Mathf.Abs(centerAngle) / 90f) : 0f;
            outcomeTargets[3] = 0f;
            outcomeTargets[4] = 1f;
        }
        else
        {
            float enemyAngle = api.Angle();
            outcomeTargets[0] = 1f;
            outcomeTargets[1] = enemyAngle > 0f ? Mathf.Clamp01(Mathf.Abs(enemyAngle) / 90f) : 0f;
            outcomeTargets[2] = enemyAngle < 0f ? Mathf.Clamp01(Mathf.Abs(enemyAngle) / 90f) : 0f;
            outcomeTargets[3] = Mathf.Abs(enemyAngle) <= CounterAngle ? 1f : 0f;
            outcomeTargets[4] = 0f;
        }

        NN.Train(previousInputs, outcomeTargets, learningRate * 1.5f);
    }

    private void ApplyNeuralOutputs(
        float[] outputs,
        float myEdge,
        float angleToEnemy,
        float distanceToEnemy,
        Vector2 predictedEnemy,
        float predictedEnemyEdge)
    {
        float accelerate = Mathf.Clamp01((outputs[0] + 1f) * 0.5f);
        float turnLeft = outputs[1];
        float turnRight = outputs[2];
        float dash = Mathf.Clamp01((outputs[3] + 1f) * 0.5f);
        float skill = Mathf.Clamp01((outputs[4] + 1f) * 0.5f);
        float absAngle = Mathf.Abs(angleToEnemy);

        if (angleToEnemy > 0f && turnLeft > nnTurnThreshold)
            TurnToward(angleToEnemy);
        else if (angleToEnemy < 0f && turnRight > nnTurnThreshold)
            TurnToward(angleToEnemy);
        else if (absAngle > FaceEnemyAngle)
            TurnToward(angleToEnemy);

        if (skill > nnSkillThreshold && ShouldStoneBlock(distanceToEnemy, myEdge))
        {
            SkillAction stoneAction = new SkillAction(InputType.Script, ActionType.SkillStone);
            if (api.CanExecute(stoneAction))
            {
                Enqueue(stoneAction);
                lastStoneTime = CurrentTime();
                counterDashUntil = CurrentTime() + CounterWindow;
            }
        }

        bool enemyNearEdge = predictedEnemyEdge >= EnemyEdgeAttack || enemyEdgeScore > 0.55f;
        float attackAngle = DynamicCounterAngle();
        float quickDashAngle = DynamicQuickDashAngle();
        float pushDuration = DynamicPushDuration(accelerate);
        float safePushLanding = DynamicSafePushLanding();
        bool pressureAllowed = accelerate > nnAccelerateThreshold || absAngle <= attackAngle || enemyNearEdge;

        bool pressureMoved = false;
        if (pressureAllowed && CanPressure(myEdge, distanceToEnemy) && absAngle <= attackAngle)
        {
            bool requireInward = myEdge > NoFlankEdge;
            if (CanTurnAndMoveSafely(angleToEnemy, pushDuration, safePushLanding, requireInward))
            {
                Enqueue(new AccelerateAction(InputType.Script, pushDuration));
                pressureMoved = true;
            }
        }

        bool dashAllowedByNN = dash > nnDashThreshold;
        bool tacticalDash = LateGameAggression() > 0.45f || enemyNearEdge || CurrentTime() <= counterDashUntil || distanceToEnemy <= StoneBlockDistance;

        if ((dashAllowedByNN || tacticalDash) && absAngle <= Mathf.Max(attackAngle, quickDashAngle) && CanAggressiveAttack(myEdge, distanceToEnemy))
        {
            if (pressureMoved)
                TryDashAfterMove(angleToEnemy, pushDuration, myEdge > NoFlankEdge);
            else
                TryDashToward(angleToEnemy);
        }
    }

    private bool OpeningCenterRush(float myEdge, float angleToCenter)
    {
        if (openingStartedAt < 0f)
            openingStartedAt = CurrentTime();

        if (myEdge > CautionEdge || ShouldCancelForOutwardVelocity(myEdge))
        {
            returningToCenter = true;
            return true;
        }

        TurnToward(angleToCenter);

        if (api.Angle(targetPos: api.BattleInfo.ArenaPosition, normalized: true) > 0.92f
            && CanTurnAndMoveSafely(angleToCenter, OpeningCenterDuration, SafeSearchLanding, true))
        {
            Enqueue(new AccelerateAction(InputType.Script, OpeningCenterDuration));
        }

        return myEdge <= SafeCenter || CurrentTime() - openingStartedAt > OpeningMaxTime;
    }

    private void LateGameFastResponse(
        float myEdge,
        float angleToCenter,
        float angleToEnemy,
        float distanceToEnemy)
    {
        bool enemyBlocksCenter = EnemyBetweenHasanAndCenterScore() > 0.45f;
        bool enemyIsClose = distanceToEnemy <= LateGameEnemyBlockDistance;

        if (myEdge <= NoFlankEdge)
        {
            FrontDash(PredictEnemyPosition(PredictionTime));
            return;
        }

        if (enemyIsClose && (enemyBlocksCenter || Mathf.Abs(angleToEnemy) <= DynamicCounterAngle()))
        {
            FaceEnemy(angleToEnemy);

            float attackDuration = DynamicPushDuration(1f);
            bool moved = false;
            if (CanTurnAndMoveSafely(angleToEnemy, attackDuration, DynamicSafePushLanding(), true))
            {
                Enqueue(new AccelerateAction(InputType.Script, attackDuration));
                moved = true;
            }

            if (Mathf.Abs(angleToEnemy) <= DynamicCounterAngle())
            {
                if (moved)
                    TryDashAfterMove(angleToEnemy, attackDuration, true);
                else
                    TryDashToward(angleToEnemy);
            }

            return;
        }

        if (Mathf.Abs(angleToCenter) > LateGameCenterHardTurnAngle)
        {
            SkillAction stoneAction = new SkillAction(InputType.Script, ActionType.SkillStone);
            if (api.CanExecute(stoneAction))
            {
                Enqueue(stoneAction);
                lastStoneTime = CurrentTime();
            }
        }

        TurnToward(angleToCenter);

        if (Mathf.Abs(angleToCenter) <= LateGameCenterHardTurnAngle
            && CanCenterRushSafely(angleToCenter, LateGameCenterRushDuration))
        {
            Enqueue(new AccelerateAction(InputType.Script, LateGameCenterRushDuration));
        }
    }

    private void RecoverToCenter(float myEdge, float angleToCenter)
    {
        if (api.CanExecute(new SkillAction(InputType.Script, ActionType.SkillStone)) && IsMovingOutward())
            Enqueue(new SkillAction(InputType.Script, ActionType.SkillStone));

        TurnToward(angleToCenter);

        if (api.Angle(targetPos: api.BattleInfo.ArenaPosition, normalized: true) > 0.96f
            && !IsMovingOutward()
            && CanTurnAndMoveSafely(angleToCenter, RecoveryDuration, SafeSearchLanding, true))
        {
            Enqueue(new AccelerateAction(InputType.Script, RecoveryDuration));
        }

        float recoveryExitEdge = Mathf.Lerp(SafeCenter, CenterControlEdge, LateGameAggression() * 0.55f);

        if (myEdge <= recoveryExitEdge && !IsMovingOutward())
            returningToCenter = false;
    }

    private void EdgeAngleLockRecover(float angleToCenter)
    {
        SkillAction stoneAction = new SkillAction(InputType.Script, ActionType.SkillStone);
        if (api.CanExecute(stoneAction))
        {
            Enqueue(stoneAction);
            lastStoneTime = CurrentTime();
        }

        TurnToward(angleToCenter);

        if (!IsFacingCenterStrongly())
            return;

        if (CanTurnAndMoveSafely(angleToCenter, EdgeLockPushDuration, SafeSearchLanding, true))
            Enqueue(new AccelerateAction(InputType.Script, EdgeLockPushDuration));
    }

    private bool ShouldStoneBlock(float distanceToEnemy, float myEdge)
    {
        if (!api.CanExecute(new SkillAction(InputType.Script, ActionType.SkillStone)))
            return false;

        if (myEdge > OutwardVelocityEdge && IsMovingOutward())
            return true;

        return distanceToEnemy <= StoneBlockDistance
            && IsEnemyChargingMe()
            && (api.EnemyRobot.LinearVelocity.magnitude >= FastEnemySpeed || enemyRushScore > 0.45f);
    }

    private TacticalState ChooseState(float myEdge, Vector2 predictedEnemy, float distanceToEnemy, EnemyAction enemyAction)
    {
        if (myEdge > CautionEdge || ShouldCancelForOutwardVelocity(myEdge))
            return TacticalState.Recover;

        float angleToPredictedEnemy = api.Angle(targetPos: predictedEnemy);
        float absEnemyAngle = Mathf.Abs(angleToPredictedEnemy);

        if (CurrentTime() <= counterDashUntil && absEnemyAngle <= CounterAngle && CanAggressiveAttack(myEdge, distanceToEnemy))
            return TacticalState.CounterDash;

        switch (enemyAction)
        {
            case EnemyAction.EdgeVulnerable:
                return CanAggressiveAttack(myEdge, distanceToEnemy) ? TacticalState.EdgeAttack : TacticalState.Search;

            case EnemyAction.Rush:
                return distanceToEnemy <= StoneBlockDistance ? TacticalState.StoneBlock : TacticalState.FrontDash;

            case EnemyAction.SideExposed:
                if (CanAggressiveAttack(myEdge, distanceToEnemy))
                    return Mathf.Abs(angleToPredictedEnemy) <= CounterAngle
                        ? TacticalState.FrontDash
                        : TacticalState.FlankDash;

                if (angleToPredictedEnemy > QuickDashAngle && angleToPredictedEnemy <= SideAttackAngle)
                    return TacticalState.TurnLeftAttack;

                if (angleToPredictedEnemy < -QuickDashAngle && angleToPredictedEnemy >= -SideAttackAngle)
                    return TacticalState.TurnRightAttack;

                return TacticalState.Search;

            case EnemyAction.Circle:
                return absEnemyAngle <= CounterAngle && CanPressure(myEdge, distanceToEnemy)
                    ? TacticalState.FrontDash
                    : TacticalState.RotateTrack;

            case EnemyAction.Retreat:
                return absEnemyAngle <= CounterAngle && CanAggressiveAttack(myEdge, distanceToEnemy)
                    ? TacticalState.FrontDash
                    : TacticalState.Search;

            default:
                return absEnemyAngle <= CounterAngle && CanPressure(myEdge, distanceToEnemy)
                    ? TacticalState.FrontDash
                    : TacticalState.Search;
        }
    }

    private EnemyAction ClassifyEnemyAction(float myEdge, Vector2 predictedEnemy, float distanceToEnemy)
    {
        float predictedEnemyEdge = EdgeRatio(predictedEnemy);

        if (predictedEnemyEdge >= EnemyEdgeAttack || enemyEdgeScore > 0.6f)
            return EnemyAction.EdgeVulnerable;

        if (ShouldStoneBlock(distanceToEnemy, myEdge)
            || (IsEnemyFrontFacingUs(distanceToEnemy) && enemyRushScore > 0.25f))
        {
            return EnemyAction.Rush;
        }

        if (EdgeRatio(api.MyRobot.Position) < NoFlankEdge
            && IsEnemyBackOrSideExposed()
            && IsFlankWorthIt(predictedEnemyEdge))
        {
            return EnemyAction.SideExposed;
        }

        if (enemyCircleScore > 0.35f || IsEnemyCircling())
            return EnemyAction.Circle;

        if (enemyRetreatScore > 0.45f)
            return EnemyAction.Retreat;

        return EnemyAction.Waiting;
    }

    private bool IsEnemySlowerOrVulnerable(float predictedEnemyEdge)
    {
        return predictedEnemyEdge >= EnemyEdgeAttack
            || enemyRetreatScore > 0.45f
            || api.EnemyRobot.LinearVelocity.magnitude <= api.MyRobot.LinearVelocity.magnitude + SlowEnemySpeed;
    }

    private bool IsFlankWorthIt(float predictedEnemyEdge)
    {
        return predictedEnemyEdge > CenterControlEdge
            || enemyCircleScore > 0.35f
            || enemyRetreatScore > 0.35f
            || enemyEdgeScore > 0.35f;
    }

    private void FrontDash(Vector2 predictedEnemy)
    {
        float angleToPredictedEnemy = api.Angle(targetPos: predictedEnemy);
        FaceEnemy(angleToPredictedEnemy);

        float duration = DynamicPushDuration(1f);
        bool requireInward = EdgeRatio(api.MyRobot.Position) > NoFlankEdge;

        bool moved = false;
        if (CanTurnAndMoveSafely(angleToPredictedEnemy, duration, DynamicSafePushLanding(), requireInward))
        {
            Enqueue(new AccelerateAction(InputType.Script, duration));
            moved = true;
        }

        if (Mathf.Abs(angleToPredictedEnemy) <= DynamicCounterAngle())
        {
            if (moved)
                TryDashAfterMove(angleToPredictedEnemy, duration, requireInward);
            else
                TryDashToward(angleToPredictedEnemy);
        }
    }

    private void EdgeAttack(Vector2 predictedEnemy, float enemyEdge, float distanceToEnemy)
    {
        float angleToPredictedEnemy = api.Angle(targetPos: predictedEnemy);
        FaceEnemy(angleToPredictedEnemy);

        float duration = DynamicPushDuration(1f);

        bool moved = false;
        if (Mathf.Abs(angleToPredictedEnemy) <= DynamicCounterAngle()
            && CanTurnAndMoveSafely(angleToPredictedEnemy, duration, DynamicSafePushLanding(), true))
        {
            Enqueue(new AccelerateAction(InputType.Script, duration));
            moved = true;
        }

        if (enemyEdge >= EnemyEdgeAttack && Mathf.Abs(angleToPredictedEnemy) <= DynamicCounterAngle() && distanceToEnemy <= CounterPushDistance)
        {
            if (moved)
                TryDashAfterMove(angleToPredictedEnemy, duration, true);
            else
                TryDashToward(angleToPredictedEnemy);
        }
    }

    private void FlankBehindDash()
    {
        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        Vector2 behindEnemy = ClampInsideArena(api.EnemyRobot.Position - enemyForward * OpeningBackDistance, SafeDashLanding);
        float angleToBehindEnemy = api.Angle(targetPos: behindEnemy);
        float distanceToBehindEnemy = Vector2.Distance(api.MyRobot.Position, behindEnemy);

        if (Mathf.Abs(api.Angle()) <= DynamicCounterAngle())
        {
            FrontDash(api.EnemyRobot.Position);
            return;
        }

        if (!IsBehindEnemy() && distanceToBehindEnemy > 0.45f)
        {
            TurnToward(angleToBehindEnemy);

            if (Mathf.Abs(angleToBehindEnemy) <= FaceEnemyAngle
                && CanTurnAndMoveSafely(angleToBehindEnemy, OpeningMoveDuration, SafePushLanding, EdgeRatio(api.MyRobot.Position) > NoFlankEdge))
            {
                Enqueue(new AccelerateAction(InputType.Script, OpeningMoveDuration));
            }

            return;
        }

        FrontDash(api.EnemyRobot.Position);
    }

    private void SideTurnAttack(TacticalState state, Vector2 predictedEnemy)
    {
        float angleToPredictedEnemy = api.Angle(targetPos: predictedEnemy);
        ActionType turnType = state == TacticalState.TurnLeftAttack ? ActionType.TurnLeft : ActionType.TurnRight;
        float turnDuration = Mathf.Clamp(Mathf.Abs(angleToPredictedEnemy) / api.MyRobot.RotateSpeed, MinTurnDuration, MaxTurnDuration);
        Enqueue(new TurnAction(InputType.Script, turnType, turnDuration));

        float duration = DynamicPushDuration(1f);

        if (Mathf.Abs(angleToPredictedEnemy) <= FaceEnemyAngle
            && CanTurnAndMoveSafely(angleToPredictedEnemy, duration, DynamicSafePushLanding(), EdgeRatio(api.MyRobot.Position) > NoFlankEdge))
        {
            Enqueue(new AccelerateAction(InputType.Script, duration));
            TryDashAfterMove(angleToPredictedEnemy, duration, EdgeRatio(api.MyRobot.Position) > NoFlankEdge);
        }
    }

    private bool IsEnemyChargingMe()
    {
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;
        if (enemyToMe.sqrMagnitude < 0.001f)
            return false;

        Vector2 enemyVelocity = api.EnemyRobot.LinearVelocity;
        if (enemyVelocity.sqrMagnitude < 0.05f)
            enemyVelocity = RotationToForward(api.EnemyRobot.Rotation) * api.EnemyRobot.MoveSpeed;

        return Vector2.Dot(enemyVelocity.normalized, enemyToMe.normalized) > 0.45f;
    }

    private bool IsEnemyFrontFacingUs(float distanceToEnemy)
    {
        if (distanceToEnemy > StoneBlockDistance)
            return false;

        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;

        if (enemyToMe.sqrMagnitude < 0.001f)
            return false;

        return Vector2.Dot(enemyForward.normalized, enemyToMe.normalized) > 0.58f;
    }

    private bool IsEnemyBackOrSideExposed()
    {
        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;

        if (enemyToMe.sqrMagnitude < 0.001f)
            return false;

        return Vector2.Dot(enemyForward.normalized, enemyToMe.normalized) < 0.35f;
    }

    private bool IsBehindEnemy()
    {
        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;

        if (enemyToMe.sqrMagnitude < 0.001f)
            return false;

        return Vector2.Dot(enemyForward.normalized, enemyToMe.normalized) < -0.45f;
    }

    private bool CanAggressiveAttack(float myEdge, float distanceToEnemy)
    {
        if (myEdge > CenterControlEdge)
            return false;

        if (IsMovingOutward() && myEdge > OutwardVelocityEdge)
            return false;

        return distanceToEnemy <= CounterPushDistance;
    }

    private bool CanPressure(float myEdge, float distanceToEnemy)
    {
        if (myEdge > CautionEdge + LateGameSafetyBonus * LateGameAggression())
            return false;

        if (IsMovingOutward() && myEdge > OutwardVelocityEdge)
            return false;

        return distanceToEnemy <= PressureDistance;
    }

    private float LateGameAggression()
    {
        float elapsed = CurrentTime();
        float duration = Mathf.Max(api.BattleInfo.Duration, 0.001f);
        float ratio = Mathf.Clamp01(elapsed / duration);

        if (ratio <= LateGameStartRatio)
            return 0f;

        return Mathf.InverseLerp(LateGameStartRatio, 1f, ratio);
    }

    private float DynamicCounterAngle()
    {
        return CounterAngle + LateGameAngleBonus * LateGameAggression();
    }

    private float DynamicQuickDashAngle()
    {
        return QuickDashAngle + LateGameAngleBonus * LateGameAggression();
    }

    private float DynamicPushDuration(float accelerate)
    {
        float baseDuration = Mathf.Lerp(PressureDuration, PushDuration, accelerate);
        return baseDuration + LateGamePushBonus * LateGameAggression();
    }

    private float DynamicSafePushLanding()
    {
        return Mathf.Min(0.93f, SafePushLanding + LateGameSafetyBonus * LateGameAggression());
    }

    private void UpdateEnemyProfile()
    {
        Vector2 enemyVelocity = api.EnemyRobot.LinearVelocity;
        if (enemyVelocity.sqrMagnitude < 0.04f)
            enemyVelocity = RotationToForward(api.EnemyRobot.Rotation) * api.EnemyRobot.MoveSpeed;

        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;
        Vector2 meToEnemy = api.EnemyRobot.Position - api.MyRobot.Position;

        float rush = 0f;
        float retreat = 0f;

        if (enemyVelocity.sqrMagnitude > 0.04f && enemyToMe.sqrMagnitude > 0.001f)
            rush = Vector2.Dot(enemyVelocity.normalized, enemyToMe.normalized) > 0.45f ? 1f : 0f;

        if (enemyVelocity.sqrMagnitude > 0.04f && meToEnemy.sqrMagnitude > 0.001f)
            retreat = Vector2.Dot(enemyVelocity.normalized, meToEnemy.normalized) > 0.45f ? 1f : 0f;

        enemyRushScore = Mathf.Lerp(enemyRushScore, rush, ProfileBlend);
        enemyCircleScore = Mathf.Lerp(enemyCircleScore, IsEnemyCircling() ? 1f : 0f, ProfileBlend);
        enemyRetreatScore = Mathf.Lerp(enemyRetreatScore, retreat, ProfileBlend);
        enemyEdgeScore = Mathf.Lerp(enemyEdgeScore, EdgeRatio(api.EnemyRobot.Position) > CenterControlEdge ? 1f : 0f, ProfileBlend);
    }

    private void SearchFromCenter(float myEdge, float angleToEnemy, float angleToCenter)
    {
        if (myEdge > CautionEdge || ShouldCancelForOutwardVelocity(myEdge))
        {
            RecoverToCenter(myEdge, angleToCenter);
            return;
        }

        FaceEnemy(angleToEnemy);

        if (TryContinuousPressure(angleToEnemy))
        {
            return;
        }

        if (Mathf.Abs(angleToEnemy) > 75f)
        {
            Enqueue(new TurnAction(InputType.Script, searchDirection > 0 ? ActionType.TurnLeft : ActionType.TurnRight, 0.16f));
        }
    }

    private bool TryContinuousPressure(float angleToEnemy)
    {
        float attackAngle = DynamicCounterAngle();
        float duration = DynamicPushDuration(0.5f);

        if (Mathf.Abs(angleToEnemy) > attackAngle)
            return false;

        if (!CanTurnAndMoveSafely(angleToEnemy, duration, DynamicSafePushLanding(), EdgeRatio(api.MyRobot.Position) > NoFlankEdge))
            return false;

        Enqueue(new AccelerateAction(InputType.Script, duration));

        if (Mathf.Abs(angleToEnemy) <= DynamicQuickDashAngle())
            TryDashAfterMove(angleToEnemy, duration, EdgeRatio(api.MyRobot.Position) > NoFlankEdge);

        return true;
    }

    private void RotateTowardEnemy(float angleToEnemy)
    {
        TurnToward(angleToEnemy);

        if (Mathf.Abs(angleToEnemy) <= FaceEnemyAngle
            && CanTurnAndMoveSafely(angleToEnemy, SearchPulseDuration, SafeSearchLanding, EdgeRatio(api.MyRobot.Position) > NoFlankEdge))
        {
            Enqueue(new AccelerateAction(InputType.Script, SearchPulseDuration));
        }
    }

    private void FaceEnemy(float angleToEnemy)
    {
        if (Mathf.Abs(angleToEnemy) > 3f)
            TurnToward(angleToEnemy);
    }

    private void TryDash()
    {
        float elapsed = api.BattleInfo.Duration - api.BattleInfo.TimeLeft;

        if (elapsed < nextDashTime)
            return;

        if (!api.CanExecute(new DashAction(InputType.Script)))
            return;

        if (!CanDashSafely())
            return;

        Enqueue(new DashAction(InputType.Script));
        lastDashTime = elapsed;
        nextDashTime = elapsed + DashInterval;
    }

    private void TryDashToward(float signedAngle)
    {
        float elapsed = api.BattleInfo.Duration - api.BattleInfo.TimeLeft;

        if (elapsed < nextDashTime)
            return;

        if (!api.CanExecute(new DashAction(InputType.Script)))
            return;

        if (!CanTurnAndDashSafely(signedAngle))
            return;

        Enqueue(new DashAction(InputType.Script));
        lastDashTime = elapsed;
        nextDashTime = elapsed + DashInterval;
    }

    private void TryDashAfterMove(float signedAngle, float moveDuration, bool requireInward)
    {
        float elapsed = api.BattleInfo.Duration - api.BattleInfo.TimeLeft;

        if (elapsed < nextDashTime)
            return;

        if (!api.CanExecute(new DashAction(InputType.Script)))
            return;

        if (!CanTurnMoveAndDashSafely(signedAngle, moveDuration, requireInward))
            return;

        Enqueue(new DashAction(InputType.Script));
        lastDashTime = elapsed;
        nextDashTime = elapsed + DashInterval;
    }

    private void TurnToward(float signedAngle)
    {
        float duration = Mathf.Clamp(
            Mathf.Abs(signedAngle) / api.MyRobot.RotateSpeed,
            MinTurnDuration,
            MaxTurnDuration);

        ActionType turnType = signedAngle > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
        Enqueue(new TurnAction(InputType.Script, turnType, duration));
    }

    private bool CanMoveSafely(float duration, float maxLandingRatio)
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new AccelerateAction(InputType.Script, duration)
        };

        var simulated = api.Simulate(actions);
        return EdgeRatio(simulated.Item1) < maxLandingRatio;
    }

    private bool CanTurnAndMoveSafely(float signedAngle, float duration, float maxLandingRatio, bool requireInward)
    {
        List<ISumoAction> actions = new List<ISumoAction>();

        if (Mathf.Abs(signedAngle) > 3f)
        {
            ActionType turnType = signedAngle > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
            float turnDuration = Mathf.Clamp(
                Mathf.Abs(signedAngle) / api.MyRobot.RotateSpeed,
                MinTurnDuration,
                MaxTurnDuration);
            actions.Add(new TurnAction(InputType.Script, turnType, turnDuration));
        }

        actions.Add(new AccelerateAction(InputType.Script, duration));

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        if (EdgeRatio(simulated.Item1) >= maxLandingRatio)
            return false;

        return !requireInward || predictedFromCenter.magnitude < currentFromCenter.magnitude - InwardProgressMargin;
    }

    private bool CanTurnAndDashSafely(float signedAngle)
    {
        List<ISumoAction> actions = new List<ISumoAction>();

        if (Mathf.Abs(signedAngle) > 3f)
        {
            ActionType turnType = signedAngle > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
            float turnDuration = Mathf.Clamp(
                Mathf.Abs(signedAngle) / api.MyRobot.RotateSpeed,
                MinTurnDuration,
                MaxTurnDuration);
            actions.Add(new TurnAction(InputType.Script, turnType, turnDuration));
        }

        actions.Add(new DashAction(InputType.Script));

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        if (EdgeRatio(simulated.Item1) >= SafeDashLanding)
            return false;

        return EdgeRatio(api.MyRobot.Position) <= NoFlankEdge
            || predictedFromCenter.magnitude < currentFromCenter.magnitude - InwardProgressMargin;
    }

    private bool CanTurnMoveAndDashSafely(float signedAngle, float moveDuration, bool requireInward)
    {
        List<ISumoAction> actions = new List<ISumoAction>();

        if (Mathf.Abs(signedAngle) > 3f)
        {
            ActionType turnType = signedAngle > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
            float turnDuration = Mathf.Clamp(
                Mathf.Abs(signedAngle) / api.MyRobot.RotateSpeed,
                MinTurnDuration,
                MaxTurnDuration);
            actions.Add(new TurnAction(InputType.Script, turnType, turnDuration));
        }

        actions.Add(new AccelerateAction(InputType.Script, moveDuration));
        actions.Add(new DashAction(InputType.Script));

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        if (EdgeRatio(simulated.Item1) >= SafeDashLanding)
            return false;

        return !requireInward || predictedFromCenter.magnitude < currentFromCenter.magnitude - InwardProgressMargin;
    }

    private bool CanCenterRushSafely(float angleToCenter, float duration)
    {
        List<ISumoAction> actions = new List<ISumoAction>();

        if (Mathf.Abs(angleToCenter) > 3f)
        {
            ActionType turnType = angleToCenter > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
            float turnDuration = Mathf.Clamp(
                Mathf.Abs(angleToCenter) / api.MyRobot.RotateSpeed,
                MinTurnDuration,
                MaxTurnDuration);
            actions.Add(new TurnAction(InputType.Script, turnType, turnDuration));
        }

        actions.Add(new AccelerateAction(InputType.Script, duration));

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        return EdgeRatio(simulated.Item1) < DynamicSafePushLanding()
            && predictedFromCenter.magnitude < currentFromCenter.magnitude - InwardProgressMargin;
    }

    private bool CanDashSafely()
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new DashAction(InputType.Script)
        };

        var simulated = api.Simulate(actions);
        return EdgeRatio(simulated.Item1) < SafeDashLanding;
    }

    private bool IsPredictedMoveInward(float duration)
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new AccelerateAction(InputType.Script, duration)
        };

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        return predictedFromCenter.magnitude < currentFromCenter.magnitude;
    }

    private bool WouldPushOutward(float duration)
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new AccelerateAction(InputType.Script, duration)
        };

        var simulated = api.Simulate(actions);
        Vector2 currentFromCenter = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 predictedFromCenter = simulated.Item1 - api.BattleInfo.ArenaPosition;

        return predictedFromCenter.magnitude > currentFromCenter.magnitude + 0.03f;
    }

    private bool ShouldCancelForOutwardVelocity(float myEdge)
    {
        return myEdge > OutwardVelocityEdge && IsMovingOutward();
    }

    private bool ShouldEdgeAngleLock(float myEdge)
    {
        if (myEdge < EdgeAngleLockEdge)
            return false;

        return !IsFacingCenterStrongly() || IsMovingOutward();
    }

    private bool IsFacingCenterStrongly()
    {
        Vector2 toCenter = api.BattleInfo.ArenaPosition - api.MyRobot.Position;
        if (toCenter.sqrMagnitude < 0.001f)
            return true;

        Vector2 forward = RotationToForward(api.MyRobot.Rotation);
        return Vector2.Dot(forward.normalized, toCenter.normalized) >= EdgeSafeFacingDot;
    }

    private bool IsMovingOutward()
    {
        Vector2 centerToRobot = api.MyRobot.Position - api.BattleInfo.ArenaPosition;
        if (centerToRobot.sqrMagnitude < 0.001f || api.MyRobot.LinearVelocity.sqrMagnitude < 0.001f)
            return false;

        return Vector2.Dot(api.MyRobot.LinearVelocity.normalized, centerToRobot.normalized) > 0.15f;
    }

    private float OutwardVelocityScore(Vector2 position, Vector2 velocity)
    {
        Vector2 centerToPosition = position - api.BattleInfo.ArenaPosition;
        if (centerToPosition.sqrMagnitude < 0.001f || velocity.sqrMagnitude < 0.001f)
            return 0f;

        return Mathf.Clamp01((Vector2.Dot(velocity.normalized, centerToPosition.normalized) + 1f) * 0.5f);
    }

    private float EnemyTowardMeScore()
    {
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;
        Vector2 enemyVelocity = api.EnemyRobot.LinearVelocity;

        if (enemyToMe.sqrMagnitude < 0.001f || enemyVelocity.sqrMagnitude < 0.001f)
            return 0f;

        return Mathf.Clamp01((Vector2.Dot(enemyVelocity.normalized, enemyToMe.normalized) + 1f) * 0.5f);
    }

    private float EnemyFacingMeScore()
    {
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;
        if (enemyToMe.sqrMagnitude < 0.001f)
            return 0f;

        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        return Mathf.Clamp01((Vector2.Dot(enemyForward.normalized, enemyToMe.normalized) + 1f) * 0.5f);
    }

    private float RelativeClosingSpeedScore()
    {
        Vector2 myToEnemy = api.EnemyRobot.Position - api.MyRobot.Position;
        if (myToEnemy.sqrMagnitude < 0.001f)
            return 0.5f;

        Vector2 relativeVelocity = api.MyRobot.LinearVelocity - api.EnemyRobot.LinearVelocity;
        float maxSpeed = Mathf.Max(api.MyRobot.MoveSpeed + api.EnemyRobot.MoveSpeed, 0.001f);
        float closing = Vector2.Dot(relativeVelocity, myToEnemy.normalized) / maxSpeed;
        return Mathf.Clamp01((closing + 1f) * 0.5f);
    }

    private float HasanBetweenEnemyAndCenterScore()
    {
        Vector2 enemyToCenter = api.BattleInfo.ArenaPosition - api.EnemyRobot.Position;
        Vector2 enemyToMe = api.MyRobot.Position - api.EnemyRobot.Position;

        if (enemyToCenter.sqrMagnitude < 0.001f || enemyToMe.sqrMagnitude < 0.001f)
            return 0f;

        return Mathf.Clamp01(Vector2.Dot(enemyToCenter.normalized, enemyToMe.normalized));
    }

    private float EnemyBetweenHasanAndCenterScore()
    {
        Vector2 meToCenter = api.BattleInfo.ArenaPosition - api.MyRobot.Position;
        Vector2 meToEnemy = api.EnemyRobot.Position - api.MyRobot.Position;

        if (meToCenter.sqrMagnitude < 0.001f || meToEnemy.sqrMagnitude < 0.001f)
            return 0f;

        return Mathf.Clamp01(Vector2.Dot(meToCenter.normalized, meToEnemy.normalized));
    }

    private float RecentCollisionScore()
    {
        float elapsed = CurrentTime() - lastCollisionTime;
        if (elapsed < 0f || elapsed > 1f)
            return 0f;

        return 1f - Mathf.Clamp01(elapsed);
    }

    private float TimeSinceScore(float lastTime, float maxSeconds)
    {
        float elapsed = CurrentTime() - lastTime;
        if (elapsed < 0f)
            return 0f;

        return Mathf.Clamp01(elapsed / maxSeconds);
    }

    private float EnemySpeedSpikeScore()
    {
        float enemySpeed = api.EnemyRobot.LinearVelocity.magnitude;
        float expectedSpeed = Mathf.Max(api.EnemyRobot.MoveSpeed, 0.001f);
        return Mathf.Clamp01(enemySpeed / expectedSpeed);
    }

    private float PredictedEdgeAfterAccelerate(float duration)
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new AccelerateAction(InputType.Script, duration)
        };

        var simulated = api.Simulate(actions);
        return EdgeRatio(simulated.Item1);
    }

    private float PredictedEdgeAfterDash()
    {
        List<ISumoAction> actions = new List<ISumoAction>
        {
            new DashAction(InputType.Script)
        };

        var simulated = api.Simulate(actions);
        return EdgeRatio(simulated.Item1);
    }

    private float EnemyFacingOutsideScore()
    {
        Vector2 centerToEnemy = api.EnemyRobot.Position - api.BattleInfo.ArenaPosition;
        if (centerToEnemy.sqrMagnitude < 0.001f)
            return 0.5f;

        Vector2 enemyForward = RotationToForward(api.EnemyRobot.Rotation);
        return Mathf.Clamp01((Vector2.Dot(enemyForward.normalized, centerToEnemy.normalized) + 1f) * 0.5f);
    }

    private bool IsEnemyCircling()
    {
        Vector2 centerToEnemy = api.EnemyRobot.Position - api.BattleInfo.ArenaPosition;
        Vector2 enemyVelocity = api.EnemyRobot.LinearVelocity;

        if (centerToEnemy.sqrMagnitude < 0.01f || enemyVelocity.sqrMagnitude < 0.04f)
            return false;

        Vector2 tangential = new Vector2(-centerToEnemy.y, centerToEnemy.x).normalized;
        float tangentMotion = Mathf.Abs(Vector2.Dot(enemyVelocity.normalized, tangential));
        float radialMotion = Mathf.Abs(Vector2.Dot(enemyVelocity.normalized, centerToEnemy.normalized));

        return tangentMotion > 0.72f && radialMotion < 0.55f;
    }

    private Vector2 PredictEnemyPosition(float secondsAhead)
    {
        Vector2 predicted = api.EnemyRobot.Position + api.EnemyRobot.LinearVelocity * secondsAhead;

        if (api.EnemyRobot.LinearVelocity.sqrMagnitude < 0.04f)
            predicted += RotationToForward(api.EnemyRobot.Rotation) * api.EnemyRobot.MoveSpeed * secondsAhead * 0.45f;

        return ClampInsideArena(predicted, 0.9f);
    }

    private Vector2 ClampInsideArena(Vector2 position, float maxRatio)
    {
        Vector2 center = api.BattleInfo.ArenaPosition;
        Vector2 offset = position - center;
        float maxDistance = api.BattleInfo.ArenaRadius * maxRatio;

        if (offset.magnitude <= maxDistance)
            return position;

        return center + offset.normalized * maxDistance;
    }

    private Vector2 RotationToForward(float rotation)
    {
        return Quaternion.Euler(0f, 0f, rotation) * Vector2.up;
    }

    private float EdgeRatio(Vector2 position)
    {
        return Vector2.Distance(position, api.BattleInfo.ArenaPosition) / api.BattleInfo.ArenaRadius;
    }

    private float CurrentTime()
    {
        return api.BattleInfo.Duration - api.BattleInfo.TimeLeft;
    }
}
