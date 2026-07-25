namespace SumoBot.Graph
{
    /// <summary>
    /// The read-only world view the interpreter needs to evaluate sensor modules, abstracted away
    /// from the runtime SumoAPI so <see cref="GraphInterpreter"/> stays pure and unit-testable
    /// (tests supply a fake). The runtime adapter (SumoApiSensors, in the game assembly) fills
    /// these from SumoAPI each tick.
    /// </summary>
    public interface IBotSensors
    {
        /// <summary>Normalized distance to the enemy (0 = touching, 1 = arena diameter apart).</summary>
        float EnemyDistance { get; }

        /// <summary>Facing alignment to the enemy as a cosine (1 = dead ahead, -1 = directly behind).</summary>
        float EnemyAngle { get; }

        /// <summary>How close I am to the arena edge (0 = center, 1 = at the rim).</summary>
        float EdgeProximity { get; }

        bool DashReady { get; }
        bool SkillReady { get; }
        bool SelfOutOfArena { get; }
        bool EnemyOutOfArena { get; }
    }
}
