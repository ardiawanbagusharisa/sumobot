Shader "Custom/Draw"
{
    Properties
    {
        _MainTex ("Canvas", 2D) = "white" {}
        _BrushColour ("Brush Colour", Color) = (1,1,1,1)
        _BackgroundColour ("Background Colour", Color) = (0,0,0,0)
        _PreviousMousePosition ("Previous Mouse Position", Vector) = (0,0,0,0)
        _MousePosition ("Mouse Position", Vector) = (0,0,0,0)
        _RectMin ("Rect Min", Vector) = (0,0,0,0)
        _RectMax ("Rect Max", Vector) = (0,0,0,0)
        _BrushSize ("Brush Size", Float) = 10
        _WiggleSize ("Wiggle Size", Float) = 0
        _MouseDown ("Mouse Down", Float) = 0
        _IsEraser ("Is Eraser", Float) = 0
        _UseRect ("Use Rect", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            ZTest Always Cull Off ZWrite Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;

            float4 _BrushColour;
            float4 _BackgroundColour;
            float4 _PreviousMousePosition;
            float4 _MousePosition;
            float4 _RectMin;
            float4 _RectMax;
            float _BrushSize;
            float _WiggleSize;
            float _MouseDown;
            float _IsEraser;
            float _UseRect;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float rand2d(float2 n)
            {
                return frac(sin(dot(n, float2(12.9898,78.233))) * 43758.5453);
            }

            float4 frag (v2f i) : SV_Target
            {
                float4 currentColor = tex2D(_MainTex, i.uv);
                if (_MouseDown < 0.5) return currentColor;

                // Pixel position in canvas space
                float2 pixelPos = i.uv / _MainTex_TexelSize.xy;

                if (_UseRect > 0.5)
                {
                    bool inside = (pixelPos.x >= _RectMin.x && pixelPos.x <= _RectMax.x &&
                                   pixelPos.y >= _RectMin.y && pixelPos.y <= _RectMax.y);

                    if (!inside) return currentColor;

                    if (_IsEraser > 0.5)
                        return float4(_BackgroundColour.rgb, 0.0);

                    return float4(lerp(currentColor.rgb, _BrushColour.rgb, 1.0), 1.0);
                }
                else
                {
                    // Wiggle
                    float2 randVec = float2(rand2d(pixelPos), rand2d(pixelPos.yx));
                    randVec = (randVec - 0.5) * _WiggleSize;

                    float2 prevPos = _PreviousMousePosition.xy + randVec;
                    float2 currPos = _MousePosition.xy + randVec;

                    float2 seg = currPos - prevPos;
                    float segLen2 = dot(seg, seg);

                    float t = segLen2 > 0 ? clamp(dot(pixelPos - prevPos, seg) / segLen2, 0.0, 1.0) : 0.0;
                    float2 closest = prevPos + seg * t;
                    float dist = length(pixelPos - closest);

                    float cov = (dist < _BrushSize) ? 1.0 : 0.0;

                    if (_IsEraser > 0.5)
                    {
                        float3 col = lerp(currentColor.rgb, _BackgroundColour.rgb, cov);
                        float a = lerp(currentColor.a, 0.0, cov);
                        return float4(col, a);
                    }

                    float3 col = lerp(currentColor.rgb, _BrushColour.rgb, cov);
                    float a = max(currentColor.a, cov);
                    return float4(col, a);
                }
            }
            ENDCG
        }
    }
}
