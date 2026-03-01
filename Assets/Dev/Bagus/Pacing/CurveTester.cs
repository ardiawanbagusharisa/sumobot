using UnityEngine;
using System.Collections.Generic;
using System;

public class CurveTester : MonoBehaviour
{
	public List<float> Threats = new List<float>();
	public List<float> Tempos = new List<float>();
	public List<float> Overall = new List<float>();

	float t;

	void Update() {
		t += Time.deltaTime;

		float threat = Mathf.Sin(t) * 0.75f; 
		Threats.Add(threat);
		float tempo = Mathf.PerlinNoise(t, 0f) * 2f - 1f;
		Tempos.Add(tempo);
		Overall.Add(threat * 0.5f + tempo * 0.5f);

		if (Threats.Count > 500) {
			Threats.RemoveAt(0);
			Tempos.RemoveAt(0);
			Overall.RemoveAt(0);
		}
	}
}
