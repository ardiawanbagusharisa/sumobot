using UnityEngine;

namespace PacingFramework
{

	public enum PacingType
	{
		Threat,
		Tempo
	}

	public class Pacing
	{
		public float Threat;
		public float Tempo;
		public float weightThreat;
		public float weightTempo;

		public Pacing(float threat = 0f, float tempo = 0f, float weightThreat = 1f, float weightTempo = 1f) {
			this.Threat = threat;
			this.Tempo = tempo;
			this.weightThreat = weightThreat;
			this.weightTempo = weightTempo;
		}

		public Pacing Clone() {
			return new Pacing(this.Threat, this.Tempo, this.weightThreat, this.weightTempo);
		}

		public static float Normalize(float value, float min, float max) {
			return max - min != 0 ? (value - min) / (max - min) : 0f;
		}
	}
}
