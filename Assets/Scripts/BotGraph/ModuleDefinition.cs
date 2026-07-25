using System;
using System.Collections.Generic;

namespace SumoBot.Graph
{
    /// <summary>
    /// The definition ("type") of a module a player can place on the board. The fixed set of
    /// these (see <see cref="ModuleLibrary"/>) is the safe vocabulary a bot graph is built from
    /// (decision-8): a saved graph may only reference TypeIds that exist here.
    ///
    /// This is pure metadata. An Action module only NAMES what it does via <see cref="TypeId"/>
    /// (e.g. "action.dash"); mapping that to a runtime SumoAPI action is the interpreter's job
    /// (E2), which lives in the game assembly and is free to reference the runtime types this
    /// data layer deliberately does not.
    /// </summary>
    public class ModuleDefinition
    {
        public string TypeId;
        public ModuleKind Kind;
        public string DisplayName;
        public string Description;
        public IReadOnlyList<PortSpec> Inputs;
        public IReadOnlyList<PortSpec> Outputs;
        public IReadOnlyList<ParamSpec> Parameters;

        public ModuleDefinition(
            string typeId,
            ModuleKind kind,
            string displayName,
            string description,
            IReadOnlyList<PortSpec> inputs = null,
            IReadOnlyList<PortSpec> outputs = null,
            IReadOnlyList<ParamSpec> parameters = null)
        {
            TypeId = typeId;
            Kind = kind;
            DisplayName = displayName;
            Description = description;
            Inputs = inputs ?? Array.Empty<PortSpec>();
            Outputs = outputs ?? Array.Empty<PortSpec>();
            Parameters = parameters ?? Array.Empty<ParamSpec>();
        }

        /// <summary>Find a port on this module by id and direction, or null if it has none.</summary>
        public PortSpec FindPort(string portId, PortDirection direction)
        {
            var list = direction == PortDirection.In ? Inputs : Outputs;
            foreach (var port in list)
                if (port.Id == portId) return port;
            return null;
        }
    }
}
