using System.Collections.Generic;

namespace SumoBot.Graph
{
    /// <summary>
    /// The fixed set of modules a bot graph may use. This vocabulary IS the "safe representation"
    /// from decision-8 — a graph can only reference these TypeIds, so there is no free-form code
    /// to sandbox. <see cref="BuildDefault"/> is the MVP library, grounded in what SumoAPI can
    /// read and what actions a bot can issue; the interpreter (E2) must implement every TypeId
    /// listed here.
    /// </summary>
    public class ModuleLibrary
    {
        private readonly Dictionary<string, ModuleDefinition> byTypeId = new();

        public IReadOnlyCollection<ModuleDefinition> All => byTypeId.Values;

        public void Register(ModuleDefinition def) => byTypeId[def.TypeId] = def;

        public ModuleDefinition Get(string typeId)
            => typeId != null && byTypeId.TryGetValue(typeId, out var def) ? def : null;

        public bool Contains(string typeId) => typeId != null && byTypeId.ContainsKey(typeId);

        private static PortSpec In(string id, string name, PortType type, bool required = true)
            => new(id, name, type, PortDirection.In, required);

        private static PortSpec Out(string id, string name, PortType type)
            => new(id, name, type, PortDirection.Out);

        /// <summary>
        /// The MVP module set. Sensors map to SumoAPI reads; logic combines values; actions map
        /// to SumoController inputs. Operators, turn directions and skills are distinct TypeIds
        /// (e.g. logic.greaterThan, action.turnLeft, action.skillBoost) so parameters stay purely
        /// numeric — no enum/choice params needed for the MVP.
        /// </summary>
        public static ModuleLibrary BuildDefault()
        {
            var lib = new ModuleLibrary();

            // Sensors — read-only views over SumoAPI.
            lib.Register(new ModuleDefinition("sensor.enemyDistance", ModuleKind.Sensor, "Enemy distance",
                "Normalized distance to the enemy (0 = touching, 1 = arena diameter apart).",
                outputs: new[] { Out("value", "Distance", PortType.Number) }));
            lib.Register(new ModuleDefinition("sensor.enemyAngle", ModuleKind.Sensor, "Enemy angle",
                "Facing alignment to the enemy as a cosine (1 = dead ahead, -1 = directly behind).",
                outputs: new[] { Out("value", "Angle", PortType.Number) }));
            lib.Register(new ModuleDefinition("sensor.edgeProximity", ModuleKind.Sensor, "Edge proximity",
                "How close I am to the arena edge (0 = center, 1 = at the rim).",
                outputs: new[] { Out("value", "Proximity", PortType.Number) }));
            lib.Register(new ModuleDefinition("sensor.dashReady", ModuleKind.Sensor, "Dash ready",
                "True when my dash is off cooldown.",
                outputs: new[] { Out("value", "Ready", PortType.Bool) }));
            lib.Register(new ModuleDefinition("sensor.skillReady", ModuleKind.Sensor, "Skill ready",
                "True when my skill is off cooldown.",
                outputs: new[] { Out("value", "Ready", PortType.Bool) }));
            lib.Register(new ModuleDefinition("sensor.selfOutOfArena", ModuleKind.Sensor, "I am falling",
                "True when I am outside the arena.",
                outputs: new[] { Out("value", "Value", PortType.Bool) }));
            lib.Register(new ModuleDefinition("sensor.enemyOutOfArena", ModuleKind.Sensor, "Enemy falling",
                "True when the enemy is outside the arena.",
                outputs: new[] { Out("value", "Value", PortType.Bool) }));

            // Logic — combine values into a decision.
            lib.Register(new ModuleDefinition("logic.constant", ModuleKind.Logic, "Constant",
                "A fixed number you choose.",
                outputs: new[] { Out("value", "Value", PortType.Number) },
                parameters: new[] { new ParamSpec("value", "Value", PortType.Number, 0.5f, -1f, 1f) }));
            lib.Register(new ModuleDefinition("logic.greaterThan", ModuleKind.Logic, "Greater than",
                "True when A is greater than B.",
                inputs: new[] { In("a", "A", PortType.Number), In("b", "B", PortType.Number) },
                outputs: new[] { Out("value", "A > B", PortType.Bool) }));
            lib.Register(new ModuleDefinition("logic.lessThan", ModuleKind.Logic, "Less than",
                "True when A is less than B.",
                inputs: new[] { In("a", "A", PortType.Number), In("b", "B", PortType.Number) },
                outputs: new[] { Out("value", "A < B", PortType.Bool) }));
            lib.Register(new ModuleDefinition("logic.and", ModuleKind.Logic, "And",
                "True when both inputs are true.",
                inputs: new[] { In("a", "A", PortType.Bool), In("b", "B", PortType.Bool) },
                outputs: new[] { Out("value", "A and B", PortType.Bool) }));
            lib.Register(new ModuleDefinition("logic.or", ModuleKind.Logic, "Or",
                "True when either input is true.",
                inputs: new[] { In("a", "A", PortType.Bool), In("b", "B", PortType.Bool) },
                outputs: new[] { Out("value", "A or B", PortType.Bool) }));
            lib.Register(new ModuleDefinition("logic.not", ModuleKind.Logic, "Not",
                "Inverts a true/false value.",
                inputs: new[] { In("a", "A", PortType.Bool) },
                outputs: new[] { Out("value", "not A", PortType.Bool) }));

            // Actions — fire when the "when" trigger is true. Duration params are in seconds
            // (min 0.1 mirrors ISumoAction.MinDuration).
            lib.Register(new ModuleDefinition("action.accelerate", ModuleKind.Action, "Accelerate",
                "Drive forward while the trigger is true.",
                inputs: new[] { In("when", "When", PortType.Bool) },
                parameters: new[] { new ParamSpec("duration", "Duration", PortType.Number, 0.2f, 0.1f, 2f) }));
            lib.Register(new ModuleDefinition("action.turnLeft", ModuleKind.Action, "Turn left",
                "Rotate left while the trigger is true.",
                inputs: new[] { In("when", "When", PortType.Bool) },
                parameters: new[] { new ParamSpec("duration", "Duration", PortType.Number, 0.2f, 0.1f, 2f) }));
            lib.Register(new ModuleDefinition("action.turnRight", ModuleKind.Action, "Turn right",
                "Rotate right while the trigger is true.",
                inputs: new[] { In("when", "When", PortType.Bool) },
                parameters: new[] { new ParamSpec("duration", "Duration", PortType.Number, 0.2f, 0.1f, 2f) }));
            lib.Register(new ModuleDefinition("action.dash", ModuleKind.Action, "Dash",
                "Dash forward when the trigger is true (if off cooldown).",
                inputs: new[] { In("when", "When", PortType.Bool) }));
            lib.Register(new ModuleDefinition("action.skillBoost", ModuleKind.Action, "Boost skill",
                "Activate the boost skill when the trigger is true.",
                inputs: new[] { In("when", "When", PortType.Bool) }));
            lib.Register(new ModuleDefinition("action.skillStone", ModuleKind.Action, "Stone skill",
                "Activate the stone skill when the trigger is true.",
                inputs: new[] { In("when", "When", PortType.Bool) }));

            return lib;
        }
    }
}
