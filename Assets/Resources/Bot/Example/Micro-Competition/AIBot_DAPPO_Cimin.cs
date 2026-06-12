using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public class AIBot_DAPPO_cimin : Bot
    {
        public override string ID => "DAPPO_Cimin";
        public override SkillType DefaultSkillType => SkillType.Boost;

        private SumoAPI api;
        private DAPPOEngine dappo;

        private const int InputSize = 7;    
        private const int ActionSize = 5;   
        private const float Gamma = 0.99f;  
        private const float K_Epochs = 3;    
        private const float Epsilon = 0.2f;  
        private const int UpdateInterval = 256; 

        private List<DAPPOExperience> trajectory = new List<DAPPOExperience>();
        
        private float frameRewardOffense = 0f;
        private float frameRewardDefense = 0f;
        private float frameRewardEscape = 0f; 
        
        private int currentFrame = 0;
        
        private Vector2 lastPos;
        private Vector2 lastEnemyPos;
        private float currentVelocity = 0f;
        private float enemyVelocity = 0f;
        private bool potentialKillShot = false;

        private float lastDistanceToEnemy = 1.0f;
        
        // State tracking for Escape & Counter system
        private bool wasInDangerZone = false;
        private int counterAttackWindow = 0;
        private int retreatTimer = 0; // NEW: Timer for Hit-and-Run tactical retreat

        public float attackAngle = 15f;       
        public float cautionZone = 0.70f;     
        public float reflexDistance = 0.35f;  
        public float baitDistance = 0.60f;    

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            dappo = new DAPPOEngine(InputSize, 32, ActionSize); 
            lastPos = api.MyRobot.Position;
            lastEnemyPos = api.EnemyRobot.Position;
            lastDistanceToEnemy = api.DistanceNormalized();
            
            currentFrame = 0;
            wasInDangerZone = false;
            counterAttackWindow = 0;
            retreatTimer = 0;
            
            Logger.Info("DAPPO Engine Initialized. Hit-and-Run Tactical Retreat Active.");
        }

        public override void OnBotUpdate()
        {
            if (api.BattleInfo.CurrentState == BattleState.Battle_End) return;

            currentFrame++; 
            if (retreatTimer > 0) retreatTimer--; // Countdown the retreat

            float dt = Time.fixedDeltaTime > 0 ? Time.fixedDeltaTime : 0.02f;
            
            Vector2 myMoveDir = (api.MyRobot.Position - lastPos);
            currentVelocity = myMoveDir.magnitude / dt;
            lastPos = api.MyRobot.Position;

            Vector2 enemyMoveDir = (api.EnemyRobot.Position - lastEnemyPos);
            enemyVelocity = enemyMoveDir.magnitude / dt;
            lastEnemyPos = api.EnemyRobot.Position;

            float currentDistance = api.DistanceNormalized();
            float distanceDelta = lastDistanceToEnemy - currentDistance;
            lastDistanceToEnemy = currentDistance;

            float safeRadius = Mathf.Max(api.BattleInfo.ArenaRadius, 0.1f);
            float distToCenterNorm = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude / safeRadius;
            
            bool inDangerZone = distToCenterNorm > 0.70f; 

            if (wasInDangerZone && !inDangerZone)
            {
                frameRewardEscape += 40.0f; 
                counterAttackWindow = 50;   
            }

            if (counterAttackWindow > 0) 
            {
                counterAttackWindow--; 
            }

            wasInDangerZone = inDangerZone;

            float[] state = GetState();
            float[] actions = dappo.GetAction(state, out float[] chosenProbs); 

            InterpretActions(actions, distToCenterNorm);
            
            (float rOffense, float rDefense) = CalculateDualRewards(actions, distanceDelta, distToCenterNorm);

            trajectory.Add(new DAPPOExperience(state, actions, chosenProbs, rOffense, rDefense));

            if (trajectory.Count >= UpdateInterval)
            {
                dappo.Update(trajectory, Gamma, Epsilon, K_Epochs);
                trajectory.Clear();
            }

            Submit();
        }

        private float[] GetState()
        {
            float safeRadius = Mathf.Max(api.BattleInfo.ArenaRadius, 0.1f);
            
            float posX = api.MyRobot.Position.x / safeRadius;
            float posY = api.MyRobot.Position.y / safeRadius;
            float angleNorm = api.Angle() / 180f; 
            float distanceNormalized = api.DistanceNormalized();
            float isDashCD = api.MyRobot.IsDashOnCooldown ? 1f : 0f;
            float isSkillCD = api.MyRobot.Skill.IsSkillOnCooldown ? 1f : 0f;
            float velNorm = Mathf.Clamp01(currentVelocity / 3.0f);

            return new float[] { posX, posY, angleNorm, distanceNormalized, isDashCD, isSkillCD, velNorm };
        }

        private void InterpretActions(float[] outputs, float distToCenterNorm)
        {
            float rawAngle = api.Angle();
            float absAngle = Mathf.Abs(rawAngle);
            float distanceToEnemyNorm = api.DistanceNormalized();

            bool isTacticalRetreat = retreatTimer > 0;
            bool needsCenterReturn = isTacticalRetreat || (distToCenterNorm > 0.70f && distToCenterNorm <= 0.88f);
            bool needsBrace = distToCenterNorm > 0.88f && !isTacticalRetreat;

            if (needsCenterReturn)
            {
                var zRot = api.MyRobot.Rotation % 360f;
                if (zRot < 0) zRot += 360f;
                Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up; 
                Vector2 meToCenter = (api.BattleInfo.ArenaPosition - api.MyRobot.Position).normalized;
                
                float angleToCenter = Vector2.SignedAngle(facingDir, meToCenter);
                float absAngleToCenter = Mathf.Abs(angleToCenter);

                if (absAngleToCenter > 5f) 
                {
                    float escapeTurnSpeed = Mathf.Clamp(absAngleToCenter / 15f, 0.7f, 1.0f);
                    if (angleToCenter > 0) 
                        Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, escapeTurnSpeed));
                    else 
                        Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, escapeTurnSpeed));
                }

                if (absAngleToCenter <= 60f)
                {
                    Enqueue(new AccelerateAction(InputType.Script, 3.0f));
                    
                    if (!api.MyRobot.IsDashOnCooldown && absAngleToCenter <= 15f)
                    {
                        Enqueue(new DashAction(InputType.Script));
                    }
                }
                
                return; 
            }

            if (needsBrace)
            {
                float braceTurnSpeed = 1.0f; 
                if (absAngle > 2f) 
                {
                    if (rawAngle > 0) Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, braceTurnSpeed));
                    else Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, braceTurnSpeed));
                }

                if (absAngle <= 45f)
                {
                    Enqueue(new AccelerateAction(InputType.Script, 3.0f));
                    
                    if (!api.MyRobot.IsDashOnCooldown)
                        Enqueue(new DashAction(InputType.Script));
                        
                    if (!api.MyRobot.Skill.IsSkillOnCooldown && absAngle <= 20f)
                        Enqueue(new SkillAction(InputType.Script));
                }
                
                return; 
            }
            bool wantsToAccelerate = outputs[0] > 0.5f;      
            bool wantsToDash = outputs[3] > 0.5f;          
            bool wantsToSkill = outputs[4] > 0.5f;         

            float turnSpeed = Mathf.Clamp(absAngle / 10f, 0.7f, 1.0f); 
            if (absAngle > 2f) 
            {
                if (rawAngle > 0) Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, turnSpeed));
                else Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, turnSpeed));
            }

            bool isFacing = absAngle <= attackAngle;
            bool panicReflex = (distanceToEnemyNorm <= reflexDistance) && isFacing;
            bool baitingReflex = (distanceToEnemyNorm > reflexDistance && distanceToEnemyNorm <= baitDistance) && isFacing;

            float speed = 0.0f; 
            if (isFacing) 
            {
                if (panicReflex) speed = 3.0f; 
                else if (baitingReflex) speed = Mathf.Abs(Mathf.Sin(Time.time * 15f) * 1.2f); 
                else if (wantsToAccelerate) speed = 2.5f; 
            }

            if (speed >= 0.1f) 
            {
                Enqueue(new AccelerateAction(InputType.Script, speed));
            }

            if (!api.MyRobot.IsDashOnCooldown && absAngle <= 12f)  
            {
                if ((wantsToDash && !baitingReflex) || panicReflex)
                    Enqueue(new DashAction(InputType.Script));
            }
            if (!api.MyRobot.Skill.IsSkillOnCooldown && absAngle <= 20f) 
            {
                if ((wantsToSkill && !baitingReflex) || panicReflex)
                    Enqueue(new SkillAction(InputType.Script));
            }
        }

        public override void OnBotCollision(BounceEvent b) 
        { 
            if (api.BattleInfo.CurrentState == BattleState.Battle_End) return;

            if (b.Actor == api.MyRobot.Side) 
            {
                float absAngle = Mathf.Abs(api.Angle());
                float totalImpactSpeed = currentVelocity + enemyVelocity;
                
                Vector2 myHeading = (api.MyRobot.Position - lastPos).normalized;
                Vector2 enemyHeading = (api.EnemyRobot.Position - lastEnemyPos).normalized;
                float headOnScore = Vector2.Dot(myHeading, enemyHeading);

                float safeRadius = Mathf.Max(api.BattleInfo.ArenaRadius, 0.1f);
                float collisionDistToCenterNorm = api.Distance(api.MyRobot.Position, api.BattleInfo.ArenaPosition).magnitude / safeRadius;

                if (collisionDistToCenterNorm > 0.70f)
                {
                    retreatTimer = 40; 
                }
                // ---------------------------------

                float hitReward = 15.0f;

                if (headOnScore < -0.4f && totalImpactSpeed > 2.5f)
                {
                    hitReward += 60.0f; 
                    potentialKillShot = true;
                }

                if (absAngle > 35f && absAngle < 95f)
                {
                    hitReward *= 2.0f; 
                }

                // COUNTER-ATTACK REWARD
                if (counterAttackWindow > 0)
                {
                    hitReward *= 2.5f;   
                    hitReward += 100.0f; 
                    counterAttackWindow = 0; 
                }

                frameRewardOffense += (hitReward * Mathf.Max(1f, currentVelocity));
            } 
            else 
            {
                frameRewardDefense -= 20.0f; 
                potentialKillShot = false;
            }
        }

        private (float, float) CalculateDualRewards(float[] actions, float distanceDelta, float distToCenterNorm)
        {
            float offense = 0f;
            float defense = 0.01f; 

            float survivalBonus = 1.0f + (currentFrame / 400f); 
            float absAngle = Mathf.Abs(api.Angle());

            var centerToMe = api.Distance(targetPos: api.MyRobot.Position, oriPos: api.BattleInfo.ArenaPosition).normalized;
            var zRot = api.MyRobot.Rotation % 360f;
            if (zRot < 0) zRot += 360f;
            Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;
            
            float facingToOutside = Vector2.Dot(facingDir, centerToMe); 
            
            // The Anti-Suicide Guardrail
            if (distToCenterNorm > 0.70f && facingToOutside > 0.3f && actions[0] > 0.5f)
            {
                defense -= (20.0f * facingToOutside); 
            }
            else if (distToCenterNorm > 0.70f && facingToOutside < -0.5f)
            {
                defense += 2.0f;
            }

            float angleAlignment = Mathf.Cos(api.Angle() * Mathf.Deg2Rad);
            offense += (angleAlignment * 2.5f);

            if (distanceDelta > 0 && angleAlignment > 0.5f) 
            {
                offense += (distanceDelta * 150.0f); 
            }

            if (absAngle <= attackAngle)
            {
                offense += (3.0f * survivalBonus); 
            }
            else if (actions[0] > 0.5f && retreatTimer == 0) 
            {
                if (angleAlignment < 0f) offense -= 5.0f; 
                else offense -= 1.0f; 
            }

            if (distToCenterNorm > 0.8f)
            {
                defense -= 5.0f; 
            }

            offense += frameRewardEscape;
            offense += frameRewardOffense;
            defense += frameRewardDefense;
            
            frameRewardEscape = 0f;
            frameRewardOffense = 0f; 
            frameRewardDefense = 0f;

            if (api.BattleInfo.CurrentState == BattleState.Battle_End)
            {
                float enemyDist = api.Distance(api.EnemyRobot.Position, api.BattleInfo.ArenaPosition).magnitude;
                float safeRadius = Mathf.Max(api.BattleInfo.ArenaRadius, 0.1f);
                
                if (enemyDist > safeRadius) 
                {
                    float winBonus = potentialKillShot ? 600.0f : 250.0f;
                    offense += winBonus; 
                }
                else if (distToCenterNorm > 1.0f) 
                {
                    defense -= 150.0f; 
                }
            }

            return (offense, defense);
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner) 
        { 
            if (state == BattleState.Battle_End) 
            {
                trajectory.Clear(); 
                ClearCommands();
                currentVelocity = 0f;
                potentialKillShot = false;
                currentFrame = 0; 
                
                lastDistanceToEnemy = 1.0f;
                wasInDangerZone = false;
                counterAttackWindow = 0;
                retreatTimer = 0;
            }
        }
    }

    // --- DAPPO & Networks ---
    public struct DAPPOExperience
    {
        public float[] State, Actions, Probs;
        public float RewardOffense, RewardDefense;

        public DAPPOExperience(float[] s, float[] a, float[] p, float rOff, float rDef) 
        { 
            State = s; 
            Actions = a; 
            Probs = p; 
            RewardOffense = rOff; 
            RewardDefense = rDef; 
        }
    }

    public class DAPPOEngine
    {
        private DAPPONet actorOffense;
        private DAPPONet actorDefense;
        private DAPPONet sharedCritic;

        public DAPPOEngine(int ins, int h, int outs)
        {
            actorOffense = new DAPPONet(ins, h, outs);
            actorDefense = new DAPPONet(ins, h, outs);
            sharedCritic = new DAPPONet(ins, h, 1); 
        }

        public float[] GetAction(float[] state, out float[] chosenProbs)
        {
            float[] rawOffense = actorOffense.Forward(state);
            float[] rawDefense = actorDefense.Forward(state);
            
            float[] chosenActions = new float[rawOffense.Length];
            chosenProbs = new float[rawOffense.Length];

            float meanProbO = 0f;
            float meanProbD = 0f;

            for (int i = 0; i < rawOffense.Length; i++)
            {
                float pO = 1f / (1f + Mathf.Exp(-rawOffense[i]));
                float pD = 1f / (1f + Mathf.Exp(-rawDefense[i]));
                meanProbO += pO;
                meanProbD += pD;
            }
            meanProbO /= rawOffense.Length;
            meanProbD /= rawDefense.Length;

            bool useOffenseActions = meanProbO < meanProbD;

            for (int i = 0; i < rawOffense.Length; i++)
            {
                float pO = 1f / (1f + Mathf.Exp(-rawOffense[i]));
                float pD = 1f / (1f + Mathf.Exp(-rawDefense[i]));

                chosenProbs[i] = Mathf.Min(pO, pD);

                float activeProb = useOffenseActions ? pO : pD;
                chosenActions[i] = UnityEngine.Random.value <= activeProb ? 1.0f : 0.0f;
            }

            return chosenActions;
        }

        public void Update(List<DAPPOExperience> batch, float gamma, float epsilon, float epochs)
        {
            for (int e = 0; e < epochs; e++)
            {
                foreach (var exp in batch)
                {
                    float currentV = sharedCritic.Forward(exp.State)[0];
                    
                    float advOffense = exp.RewardOffense - currentV;
                    float advDefense = exp.RewardDefense - currentV;

                    sharedCritic.Backward(exp.State, new float[] { advOffense + advDefense });

                    float[] newRawOffense = actorOffense.Forward(exp.State);
                    float[] newRawDefense = actorDefense.Forward(exp.State);

                    float[] errOffense = new float[exp.Actions.Length];
                    float[] errDefense = new float[exp.Actions.Length];

                    for (int i = 0; i < exp.Actions.Length; i++)
                    {
                        float newProbO = 1f / (1f + Mathf.Exp(-newRawOffense[i]));
                        float newProbD = 1f / (1f + Mathf.Exp(-newRawDefense[i]));

                        float minNewProb = Mathf.Min(newProbO, newProbD);
                        
                        float ratio = Mathf.Clamp(minNewProb, 0.01f, 1.0f) / Mathf.Max(exp.Probs[i], 0.01f);

                        float surr1O = ratio * advOffense;
                        float surr2O = Mathf.Clamp(ratio, 1f - epsilon, 1f + epsilon) * advOffense;
                        errOffense[i] = -Math.Min(surr1O, surr2O);

                        float surr1D = ratio * advDefense;
                        float surr2D = Mathf.Clamp(ratio, 1f - epsilon, 1f + epsilon) * advDefense;
                        errDefense[i] = -Math.Min(surr1D, surr2D);
                    }

                    actorOffense.Backward(exp.State, errOffense);
                    actorDefense.Backward(exp.State, errDefense);
                }
            }
        }
    }

    public class DAPPONet
    {
        public float[,] weights;
        public float[] bias;
        private int iSize, oSize;
        
        public DAPPONet(int i, int h, int o)
        {
            iSize = i; oSize = o;
            weights = new float[i, o]; 
            bias = new float[o];
            for (int x = 0; x < i; x++)
                for (int y = 0; y < o; y++)
                    weights[x, y] = UnityEngine.Random.Range(-0.1f, 0.1f); 
        }
        
        public float[] Forward(float[] input)
        {
            float[] output = new float[oSize];
            for (int j = 0; j < oSize; j++)
            {
                output[j] = bias[j];
                for (int k = 0; k < iSize; k++)
                    output[j] += input[k] * weights[k, j];
                if (float.IsNaN(output[j])) output[j] = 0f; 
            }
            return output;
        }
        
        public void Backward(float[] input, float[] errors)
        {
            float lr = 0.0015f; 
            float weightDecay = 0.9995f; 
            for (int j = 0; j < oSize; j++)
            {
                float clampedError = Mathf.Clamp(errors[j], -1.0f, 1.0f);
                for (int k = 0; k < iSize; k++)
                {
                    weights[k, j] *= weightDecay;
                    weights[k, j] -= lr * clampedError * input[k];
                }
                bias[j] -= lr * clampedError;
            }
        }
    }
}