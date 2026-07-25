using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SumoServices.Tests
{
    /// <summary>
    /// In-memory ICatalogService for unit tests: seeded with items in the constructor, no
    /// Resources / JSON. LoadAsync is a no-op success.
    /// </summary>
    public class FakeCatalogService : ICatalogService
    {
        private readonly Dictionary<string, CatalogItem> byId = new();

        public FakeCatalogService(params CatalogItem[] items)
        {
            foreach (var item in items) byId[item.Id] = item;
        }

        public IReadOnlyList<CatalogItem> AllItems => byId.Values.ToList();

        public Task<ServiceResult> LoadAsync() => Task.FromResult(ServiceResult.Ok());

        public CatalogItem GetById(string itemId)
            => itemId != null && byId.TryGetValue(itemId, out var item) ? item : null;
    }
}
