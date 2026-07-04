#pragma once
#include "Math.h"

namespace ed {

// Orbital camera around a target point.
struct Camera {
    V3 target{0, 1.0f, 0};
    float yaw = 0.6f;      // radians
    float pitch = 0.15f;
    float dist = 3.5f;
    float fovy = 0.8f;     // radians (~45deg)
    float znear = 0.01f, zfar = 100.0f;

    V3 eye() const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw), sy = std::sin(yaw);
        return { target.x + dist * cp * sy,
                 target.y + dist * sp,
                 target.z + dist * cp * cy };
    }
    M4 view() const { return lookAt(eye(), target, {0, 1, 0}); }
    M4 proj(float aspect) const { return perspective(fovy, aspect, znear, zfar); }

    void orbit(float dx, float dy) {
        yaw   -= dx * 0.01f;
        pitch += dy * 0.01f;
        const float lim = 1.55f;
        if (pitch >  lim) pitch =  lim;
        if (pitch < -lim) pitch = -lim;
    }
    void pan(float dx, float dy) {
        float cy = std::cos(yaw), sy = std::sin(yaw);
        V3 right{cy, 0, -sy};
        V3 up{0, 1, 0};
        float s = dist * 0.0015f;
        target = target - right * (dx * s) + up * (dy * s);
    }
    void zoom(float d) {
        dist *= std::pow(1.1f, -d);
        if (dist < 0.1f) dist = 0.1f;
        if (dist > 50.0f) dist = 50.0f;
    }
    // Build a world-space ray from normalized device coords (nx,ny in [-1,1]).
    Ray rayFromNDC(float nx, float ny, float aspect) const {
        // reconstruct via inverse of proj*view applied to near/far points — do it simply:
        V3 e = eye();
        V3 f = normalize(target - e);
        V3 r = normalize(cross(f, {0,1,0}));
        V3 u = cross(r, f);
        float th = std::tan(fovy * 0.5f);
        V3 dir = normalize(f + r * (nx * th * aspect) + u * (ny * th));
        return { e, dir };
    }
};

} // namespace ed
