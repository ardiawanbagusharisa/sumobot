using Newtonsoft.Json;

namespace SumoBot.Graph
{
    /// <summary>
    /// JSON (de)serialization for a <see cref="BotGraph"/>. The graph is plain data with no
    /// polymorphism (nodes are uniform, identified by a TypeId string), so default Newtonsoft
    /// settings round-trip cleanly — no custom converter, unlike CatalogItemConverter. This is
    /// the on-disk / over-the-wire form the editor (E3) and publish (decision-7) use.
    /// </summary>
    public static class GraphSerializer
    {
        private static readonly JsonSerializerSettings Settings = new()
        {
            Formatting = Formatting.Indented,
            NullValueHandling = NullValueHandling.Ignore,
        };

        public static string ToJson(BotGraph graph) => JsonConvert.SerializeObject(graph, Settings);

        public static BotGraph FromJson(string json) => JsonConvert.DeserializeObject<BotGraph>(json, Settings);
    }
}
