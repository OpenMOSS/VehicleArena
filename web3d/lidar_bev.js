import * as THREE from "three";

// A processed geometry display, not an optical camera or simulated ray scan.
// It shares the frozen scene with the cockpit and never advances the world.
export class LidarBEVRenderer {
  constructor(scene, renderer, surfaces, weatherGroup) {
    this.scene = scene;
    this.renderer = renderer;
    this.surfaces = surfaces;
    this.weatherGroup = weatherGroup;
    this.cache = new Map();
    this.mask = {
      bevOrigin: { value: new THREE.Vector2() },
      bevForward: { value: new THREE.Vector2() },
      bevRange: { value: 80 },
      bevHalfFovCos: { value: Math.cos(THREE.MathUtils.degToRad(80)) },
    };
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 600);
    this.ambient = new THREE.HemisphereLight(0xffffff, 0x888888, 2.2);
    this.shapeLight = new THREE.DirectionalLight(0xffffff, 2.0);
    this.background = new THREE.Color(0x383838);
    this.otherBody = this.withMask(new THREE.MeshStandardMaterial({
      color: 0xb0b0b0, roughness: 0.85, metalness: 0,
    }));
  }

  withMask(material) {
    const uniforms = this.mask;
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader.replace("#include <common>",
        "#include <common>\nvarying vec3 bevWorldPosition;").replace(
        "#include <project_vertex>",
        "#include <project_vertex>\nbevWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;");
      shader.fragmentShader = shader.fragmentShader.replace("#include <common>",
        `#include <common>
        varying vec3 bevWorldPosition;
        uniform vec2 bevOrigin;
        uniform vec2 bevForward;
        uniform float bevRange;
        uniform float bevHalfFovCos;`).replace("void main() {", `void main() {
        vec2 bevDelta = bevWorldPosition.xz - bevOrigin;
        float bevDistance = length(bevDelta);
        // Preserve an 8 m immediate ego context, including the road beneath
        // the body; beyond it apply the configured planar sensor cone.
        if (bevDistance > bevRange || (bevDistance > 8.0 &&
            dot(bevDelta, bevForward) < bevDistance * bevHalfFovCos)) discard;`);
    };
    material.customProgramCacheKey = () => "vehiclearena-processed-bev-mask-v1";
    return material;
  }

  material(original, ego = false) {
    let variants = this.cache.get(original);
    if (!variants) { variants = new Map(); this.cache.set(original, variants); }
    if (variants.has(ego)) return variants.get(ego);
    const mat = original.clone();
    let color = 0x999999;
    if (original === this.surfaces.ground) color = 0x383838;
    else if (original === this.surfaces.road || original === this.surfaces.junction) color = 0x606060;
    else if (original === this.surfaces.crosswalk || original === this.surfaces.arrow || original.isLineBasicMaterial) color = 0xbdbdbd;
    if (ego) {
      if (original.userData.bevRole === "body") color = 0x147bdd;
      else if (original.userData.bevRole === "dark") color = 0x303030;
    }
    if (mat.color) mat.color.setHex(color);
    for (const key of ["map", "emissiveMap", "lightMap", "envMap"]) if (key in mat) mat[key] = null;
    if (mat.emissive) mat.emissive.setHex(0);
    if ("emissiveIntensity" in mat) mat.emissiveIntensity = 0;
    if ("metalness" in mat) mat.metalness = 0;
    if ("roughness" in mat) mat.roughness = 0.85;
    mat.vertexColors = false;
    if (!ego) this.withMask(mat);
    mat.needsUpdate = true;
    variants.set(ego, mat);
    return mat;
  }

  reset() {
    for (const variants of this.cache.values()) for (const mat of variants.values()) mat.dispose();
    this.cache.clear();
  }

  render(focus, actors, sensor = {}) {
    const range = Number(sensor.range_m ?? 80);
    const fov = Number(sensor.horizontal_fov_deg ?? 160);
    if (!focus || focus.kind !== "vehicle") throw new Error("LiDAR BEV requires the requested focus vehicle");
    if (!Number.isFinite(range) || range < 10 || range > 300 ||
        !Number.isFinite(fov) || fov < 30 || fov > 360) throw new Error("Invalid LiDAR display range/FOV");
    const { scene, renderer } = this;
    const started = performance.now();
    const position = focus.object.position;
    const heading = focus.direction;
    this.mask.bevOrigin.value.set(position.x, position.z);
    this.mask.bevForward.value.set(heading.x, heading.z);
    this.mask.bevRange.value = range;
    this.mask.bevHalfFovCos.value = Math.cos(THREE.MathUtils.degToRad(fov / 2));
    const actorParts = new Map();
    for (const actor of actors.values()) actor.object.traverse((part) => {
      if (part.isMesh) actorParts.set(part, actor === focus);
    });
    const restore = [];
    const size = renderer.getSize(new THREE.Vector2());
    const viewport = renderer.getViewport(new THREE.Vector4());
    const scissor = renderer.getScissor(new THREE.Vector4());
    const old = {
      background: scene.background, fog: scene.fog, weather: this.weatherGroup.visible,
      shadows: renderer.shadowMap.enabled, tone: renderer.toneMapping,
      exposure: renderer.toneMappingExposure, scissorTest: renderer.getScissorTest(),
      ratio: renderer.getPixelRatio(), focusVisible: focus.object.visible,
      clearColor: renderer.getClearColor(new THREE.Color()).clone(), clearAlpha: renderer.getClearAlpha(),
    };
    try {
      scene.traverse((obj) => {
        if (obj.isLight) { const visible = obj.visible; restore.push(() => { obj.visible = visible; }); obj.visible = false; }
        if (!obj.material) return;
        const original = obj.material;
        restore.push(() => { obj.material = original; });
        if (actorParts.get(obj) === false) obj.material = this.otherBody;
        else {
          const convert = (mat) => this.material(mat, actorParts.get(obj) === true);
          obj.material = Array.isArray(original) ? original.map(convert) : convert(original);
        }
      });
      scene.background = this.background;
      scene.fog = null;
      this.weatherGroup.visible = false;
      focus.object.visible = true;
      renderer.shadowMap.enabled = false;
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.toneMappingExposure = 1;
      this.shapeLight.position.copy(position).add(new THREE.Vector3(-25, 60, 20));
      this.shapeLight.target.position.copy(position);
      scene.add(this.ambient, this.shapeLight, this.shapeLight.target);
      const elevation = THREE.MathUtils.degToRad(35);
      const distance = 20;
      const halfFov = Math.tan(THREE.MathUtils.degToRad(20));
      const centerOffset = 0.5 * halfFov * distance / (Math.sin(elevation) + 0.5 * halfFov * Math.cos(elevation));
      const center = position.clone().addScaledVector(heading, centerOffset);
      this.camera.position.copy(center).addScaledVector(heading, -distance * Math.cos(elevation));
      this.camera.position.y += distance * Math.sin(elevation);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(center);
      this.camera.updateMatrixWorld(true);
      renderer.setPixelRatio(1);
      renderer.setSize(1024, 1024, false);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, 1024, 1024);
      renderer.clear(true, true, true);
      renderer.render(scene, this.camera);
      // Capture the canvas only: no dashboard, debug labels, or DOM overlays.
      const png = renderer.domElement.toDataURL("image/png");
      const anchor = position.clone().project(this.camera);
      return { png_data_url: png, width: 1024, height: 1024,
        focus_entity_id: focus.id, ego_ndc: anchor.toArray(),
        render_ms: performance.now() - started,
        view: "web3d_ego_oblique", elevation_deg: 35, fov_deg: 40,
        sensor_range_m: range, sensor_horizontal_fov_deg: fov,
        material_cache_size: this.cache.size };
    } finally {
      restore.reverse().forEach((fn) => fn());
      scene.remove(this.ambient, this.shapeLight, this.shapeLight.target);
      scene.background = old.background;
      scene.fog = old.fog;
      this.weatherGroup.visible = old.weather;
      focus.object.visible = old.focusVisible;
      renderer.shadowMap.enabled = old.shadows;
      renderer.toneMapping = old.tone;
      renderer.toneMappingExposure = old.exposure;
      renderer.setPixelRatio(old.ratio);
      renderer.setSize(size.x, size.y, false);
      renderer.setViewport(viewport);
      renderer.setScissor(scissor);
      renderer.setScissorTest(old.scissorTest);
      renderer.setClearColor(old.clearColor, old.clearAlpha);
    }
  }
}
