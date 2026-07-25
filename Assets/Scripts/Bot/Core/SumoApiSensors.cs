using SumoBot.Graph;

namespace SumoBot
{
    /// <summary>
    /// Adapts the runtime <see cref="SumoAPI"/> into the interpreter's engine-free
    /// <see cref="IBotSensors"/> view (decision-8 E2). All reads are cheap value lookups, so a
    /// fresh instance is created each tick and the interpreter never touches Unity types.
    /// </summary>
    public class SumoApiSensors : IBotSensors
    {
        private readonly SumoAPI api;

        public SumoApiSensors(SumoAPI api)
        {
            this.api = api;
        }

        public float EnemyDistance => api.DistanceNormalized();

        public float EnemyAngle => api.Angle(normalized: true);

        public float EdgeProximity
        {
            get
            {
                float radius = api.BattleInfo.ArenaRadius;
                if (radius <= 0f) return 0f;
                return (api.MyRobot.Position - api.BattleInfo.ArenaPosition).magnitude / radius;
            }
        }

        public bool DashReady => !api.MyRobot.IsDashOnCooldown;

        public bool SkillReady => !api.MyRobot.Skill.IsSkillOnCooldown;

        public bool SelfOutOfArena => api.MyRobot.IsOutFromArena;

        public bool EnemyOutOfArena => api.EnemyRobot.IsOutFromArena;
    }
}
