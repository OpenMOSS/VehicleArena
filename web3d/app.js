import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { LidarBEVRenderer } from "./lidar_bev.js";

const viewport = document.querySelector("#viewport");
const loading = document.querySelector("#loading");
const errorBox = document.querySelector("#error");
const cockpitOverlay = document.querySelector("#cockpit");
const mapLabel = document.querySelector("#map-label");
const simTimeLabel = document.querySelector("#sim-time");
const speedLabel = document.querySelector("#ego-speed");
const connectionLabel = document.querySelector("#connection-status");
const controlSourceLabel = document.querySelector("#control-source");

const scene = new THREE.Scene();
const skyCanvas = document.createElement("canvas");
skyCanvas.width = 32;
skyCanvas.height = 256;
const skyContext = skyCanvas.getContext("2d");
const skyTexture = new THREE.CanvasTexture(skyCanvas);
skyTexture.colorSpace = THREE.SRGBColorSpace;
skyTexture.minFilter = THREE.LinearFilter;
skyTexture.magFilter = THREE.LinearFilter;
skyTexture.generateMipmaps = false;

function paintSky(zenith, horizon) {
  const gradient = skyContext.createLinearGradient(0, 0, 0, skyCanvas.height);
  gradient.addColorStop(0, `#${zenith.getHexString()}`);
  gradient.addColorStop(0.68, `#${horizon.getHexString()}`);
  gradient.addColorStop(1, `#${horizon.getHexString()}`);
  skyContext.fillStyle = gradient;
  skyContext.fillRect(0, 0, skyCanvas.width, skyCanvas.height);
  skyTexture.needsUpdate = true;
}

paintSky(new THREE.Color(0x789fb3), new THREE.Color(0xc3d2d6));
scene.background = skyTexture;
scene.fog = new THREE.FogExp2(0xc3d2d6, 0.0028);

const camera = new THREE.PerspectiveCamera(
  68,
  window.innerWidth / window.innerHeight,
  0.08,
  1200,
);
camera.position.set(0, 90, 100);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 20;
controls.maxDistance = 420;
controls.target.set(0, 0, 0);
controls.enabled = false;

scene.add(new THREE.HemisphereLight(0xdbeeff, 0x465844, 2.1));
const sun = new THREE.DirectionalLight(0xfff4dc, 3.0);
sun.position.set(-90, 145, 65);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -190;
sun.shadow.camera.right = 190;
sun.shadow.camera.top = 190;
sun.shadow.camera.bottom = -190;
sun.shadow.camera.near = 5;
sun.shadow.camera.far = 350;
scene.add(sun);

function createGroundTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  context.fillStyle = "#71836d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < 16; row += 1) {
    context.fillStyle = row % 2 === 0
      ? "rgba(190,205,178,0.035)"
      : "rgba(29,53,31,0.025)";
    context.fillRect(0, row * 16, canvas.width, 8);
  }
  for (let index = 0; index < 720; index += 1) {
    const x = deterministicUnit(index, 11) * canvas.width;
    const y = deterministicUnit(index, 12) * canvas.height;
    const light = deterministicUnit(index, 13) > 0.54;
    context.fillStyle = light
      ? "rgba(197,211,185,0.075)"
      : "rgba(27,51,30,0.07)";
    context.fillRect(x, y, 1.2, 1.2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(24, 24);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(760, 760),
  new THREE.MeshStandardMaterial({
    color: 0xb4bbaa,
    map: createGroundTexture(),
    roughness: 1,
  }),
);
const groundMaterial = ground.material;
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.055;
ground.receiveShadow = true;
scene.add(ground);

const staticGroup = new THREE.Group();
const actorGroup = new THREE.Group();
scene.add(staticGroup, actorGroup);

function deterministicUnit(index, salt) {
  const value = Math.sin(
    (index + 1) * 12.9898 + salt * 78.233,
  ) * 43758.5453;
  return value - Math.floor(value);
}

function createRainSystem(count) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(new Float32Array(count * 6), 3),
  );
  geometry.setDrawRange(0, 0);
  const material = new THREE.LineBasicMaterial({
    color: 0x72b8ff,
    transparent: true,
    opacity: 0.42,
    depthWrite: false,
  });
  const object = new THREE.LineSegments(geometry, material);
  object.frustumCulled = false;
  object.visible = false;
  object.userData.seeds = Array.from({ length: count }, (_, index) => ({
    x: deterministicUnit(index, 1),
    y: deterministicUnit(index, 2),
    z: deterministicUnit(index, 3),
  }));
  return object;
}

function createSnowSystem(count) {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(16, 16, 1, 16, 16, 15);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.58, "rgba(245,250,255,0.92)");
  gradient.addColorStop(1, "rgba(235,245,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 32, 32);
  const flakeTexture = new THREE.CanvasTexture(canvas);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(new Float32Array(count * 3), 3),
  );
  geometry.setDrawRange(0, 0);
  const material = new THREE.PointsMaterial({
    color: 0xf3f8ff,
    size: 0.2,
    map: flakeTexture,
    alphaTest: 0.04,
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const object = new THREE.Points(geometry, material);
  object.frustumCulled = false;
  object.visible = false;
  object.userData.seeds = Array.from({ length: count }, (_, index) => ({
    x: deterministicUnit(index, 4),
    y: deterministicUnit(index, 5),
    z: deterministicUnit(index, 6),
    drift: deterministicUnit(index, 7) * Math.PI * 2,
  }));
  return object;
}

const weatherGroup = new THREE.Group();
const rainSystem = createRainSystem(760);
const snowSystem = createSnowSystem(620);
weatherGroup.add(rainSystem, snowSystem);
camera.add(weatherGroup);
scene.add(camera);

const WEATHER_PROFILES = {
  sunny: { rain: 0, snow: 0, fog: 0.0028, darkening: 1 },
  clear: { rain: 0, snow: 0, fog: 0.0028, darkening: 1 },
  cloudy: { rain: 0, snow: 0, fog: 0.0028, darkening: 0.86 },
  rainy: { rain: 0.42, snow: 0, fog: 0.0035, darkening: 0.86, wet: 0.62 },
  heavy_rain: { rain: 1, snow: 0, fog: 0.0065, darkening: 0.68, wet: 1 },
  foggy: { rain: 0, snow: 0, fog: 0.014, darkening: 0.72 },
  snowy: { rain: 0, snow: 0.42, fog: 0.0038, darkening: 0.84, snowCover: 0.45 },
  heavy_snow: { rain: 0, snow: 1, fog: 0.0075, darkening: 0.7, snowCover: 1 },
  hail: { rain: 0.72, snow: 0.32, fog: 0.005, darkening: 0.7, wet: 0.9 },
};

let activeWeather = {
  name: "sunny", rain: 0, snow: 0, windSpeedMps: 0, fogDensity: 0.0022,
};

const roadMaterial = new THREE.MeshStandardMaterial({
  color: 0x303437,
  roughness: 0.92,
  metalness: 0.02,
  side: THREE.DoubleSide,
});
const junctionMaterial = roadMaterial.clone();
junctionMaterial.color.setHex(0x373b3e);
const crosswalkMaterial = new THREE.MeshStandardMaterial({
  color: 0xe9e5d7,
  roughness: 0.82,
  side: THREE.DoubleSide,
});
const groundArrowMaterial = new THREE.MeshStandardMaterial({
  color: 0xf2efe3,
  roughness: 0.78,
  side: THREE.DoubleSide,
});

function polygonGeometry(points, elevation = 0) {
  const contour = points.map(([x, z]) => new THREE.Vector2(x, z));
  const triangles = THREE.ShapeUtils.triangulateShape(contour, []);
  const positions = [];
  for (const triangle of triangles) {
    for (const index of triangle) {
      positions.push(contour[index].x, elevation, contour[index].y);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function polygonMesh(points, material, elevation) {
  const mesh = new THREE.Mesh(polygonGeometry(points, elevation), material);
  mesh.receiveShadow = true;
  return mesh;
}

function addPolyline(record) {
  if (!record.points_xz || record.points_xz.length < 2) return;
  const points = record.points_xz.map(
    ([x, z]) => new THREE.Vector3(x, 0.065 + (record.elevation_m || 0), z),
  );
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const color = new THREE.Color(record.color || "#e8e4d3");
  const defaultOpacity = record.kind === "lane_divider"
    ? 0.78
    : record.kind === "road_edge" ? 0.9 : 1;
  const opacity = Number(record.opacity ?? defaultOpacity);
  const material = record.dashed
    ? new THREE.LineDashedMaterial({
      color,
      dashSize: Number(record.dash_size_m ?? 3.4),
      gapSize: Number(record.gap_size_m ?? 5.2),
      transparent: opacity < 1,
      opacity,
    })
    : new THREE.LineBasicMaterial({
      color,
      transparent: opacity < 1,
      opacity,
    });
  const line = new THREE.Line(geometry, material);
  if (record.dashed) line.computeLineDistances();
  staticGroup.add(line);
}

function addThickSegment(points, color, width, elevation = 0.075) {
  if (!points || points.length < 2) return;
  const [a, b] = [points[0], points.at(-1)];
  const dx = b[0] - a[0];
  const dz = b[1] - a[1];
  const length = Math.hypot(dx, dz);
  if (length < 0.01) return;
  const nx = (-dz / length) * width / 2;
  const nz = (dx / length) * width / 2;
  staticGroup.add(polygonMesh([
    [a[0] + nx, a[1] + nz],
    [b[0] + nx, b[1] + nz],
    [b[0] - nx, b[1] - nz],
    [a[0] - nx, a[1] - nz],
  ], new THREE.MeshStandardMaterial({ color, roughness: 0.8 }), elevation));
}

function thickPathPolygons(points, width = 0.2) {
  const parts = [];
  for (let index = 1; index < points.length; index += 1) {
    const a = points[index - 1];
    const b = points[index];
    const dx = b[0] - a[0];
    const dz = b[1] - a[1];
    const length = Math.hypot(dx, dz);
    if (length < 0.001) continue;
    const nx = (-dz / length) * width / 2;
    const nz = (dx / length) * width / 2;
    parts.push([
      [a[0] + nx, a[1] + nz],
      [b[0] + nx, b[1] + nz],
      [b[0] - nx, b[1] - nz],
      [a[0] - nx, a[1] - nz],
    ]);
  }
  return parts;
}

function movementArrowPolygons(movement, anchorX = 0) {
  if (movement === "straight") {
    return [[
      [anchorX - 0.55, 0.45],
      [anchorX + 0.55, 0.45],
      [anchorX, 1.42],
    ]];
  }
  if (movement === "left" || movement === "right") {
    // The browser ground plane is X/Z with +Z as local forward. In this
    // right-handed frame physical left is local +X, not -X as in a 2-D image.
    const side = movement === "left" ? 1 : -1;
    const branchEnd = side * 0.95;
    return [
      [[Math.min(anchorX, branchEnd), 0.18], [Math.max(anchorX, branchEnd), 0.18],
        [Math.max(anchorX, branchEnd), 0.4], [Math.min(anchorX, branchEnd), 0.4]],
      side > 0
        ? [[1.42, 0.29], [0.72, 0.76], [0.72, -0.18]]
        : [[-1.42, 0.29], [-0.72, -0.18], [-0.72, 0.76]],
    ];
  }
  if (movement === "uturn") {
    const curve = [
      [anchorX, -1.5],
      [anchorX, 0.42],
      [anchorX + 0.08, 0.7],
      [anchorX + 0.35, 0.91],
      [anchorX + 0.73, 0.91],
      [anchorX + 1.02, 0.7],
      [anchorX + 1.12, 0.4],
      [anchorX + 1.12, -0.28],
    ];
    const parts = thickPathPolygons(curve);
    const returnX = anchorX + 1.12;
    parts.push([
      [returnX, -1.02],
      [returnX - 0.45, -0.22],
      [returnX + 0.45, -0.22],
    ]);
    return parts;
  }
  return [];
}

function addGroundArrow(record) {
  const movements = record.movements || [];
  if (!movements.length) return;
  const root = new THREE.Group();
  root.position.set(
    record.position_xz[0],
    Number(record.elevation_m || 0),
    record.position_xz[1],
  );
  root.rotation.y = Number(record.rotation_y_rad || 0);
  const hasUturn = movements.includes("uturn");
  // A standard U-turn marking keeps the forward stem on the physical right
  // and bends over to the physical left. In local X/Z those are -X and +X.
  const anchorX = hasUturn ? -0.34 : 0;
  if (!hasUturn) {
    const shaft = [
      [anchorX - 0.11, -1.5], [anchorX + 0.11, -1.5],
      [anchorX + 0.11, 0.55], [anchorX - 0.11, 0.55],
    ];
    root.add(polygonMesh(shaft, groundArrowMaterial, 0.09));
  }
  movements.forEach((movement) => {
    for (const polygon of movementArrowPolygons(movement, anchorX)) {
      root.add(polygonMesh(polygon, groundArrowMaterial, 0.09));
    }
  });
  staticGroup.add(root);
}

function createBuildingFacadeTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  context.fillStyle = "#aeb6b8";
  context.fillRect(0, 0, 96, 96);
  context.fillStyle = "#526a75";
  context.fillRect(10, 11, 28, 20);
  context.fillRect(58, 11, 28, 20);
  context.fillRect(10, 55, 28, 20);
  context.fillRect(58, 55, 28, 20);
  context.fillStyle = "rgba(220,235,238,0.18)";
  context.fillRect(12, 13, 24, 3);
  context.fillRect(60, 13, 24, 3);
  context.fillRect(12, 57, 24, 3);
  context.fillRect(60, 57, 24, 3);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

const buildingFacadeTexture = createBuildingFacadeTexture();
const buildingRoofMaterial = new THREE.MeshStandardMaterial({
  color: 0x35444b,
  roughness: 0.86,
});
const buildingFoundationMaterial = new THREE.MeshStandardMaterial({
  color: 0x7a8382,
  roughness: 0.96,
});

function addBuilding(building) {
  const [width, height, depth] = building.dimensions_m;
  const facadeTexture = buildingFacadeTexture.clone();
  facadeTexture.needsUpdate = true;
  facadeTexture.repeat.set(
    Math.max(2, Math.round(width / 6)),
    Math.max(3, Math.round(height / 4)),
  );
  const facadeColor = new THREE.Color(building.color).lerp(
    new THREE.Color(0x87969d), 0.42,
  );
  const facadeMaterial = new THREE.MeshStandardMaterial({
    color: facadeColor,
    map: facadeTexture,
    roughness: 0.72,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    [
      facadeMaterial, facadeMaterial,
      buildingRoofMaterial, buildingFoundationMaterial,
      facadeMaterial, facadeMaterial,
    ],
  );
  mesh.position.set(building.position_xz[0], height / 2, building.position_xz[1]);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  staticGroup.add(mesh);

  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.5, 0.22, depth + 0.5),
    buildingRoofMaterial,
  );
  roof.position.copy(mesh.position);
  roof.position.y = height + 0.12;
  staticGroup.add(roof);

  const foundation = new THREE.Mesh(
    new THREE.BoxGeometry(width + 1.4, 0.16, depth + 1.4),
    buildingFoundationMaterial,
  );
  foundation.position.set(
    building.position_xz[0], 0.03, building.position_xz[1],
  );
  foundation.receiveShadow = true;
  staticGroup.add(foundation);
}

const signalControllers = [];

function signalArrowGeometry(turn) {
  // Flat front-facing lenses preserve direction visibility: no glowing bulbs
  // on the back of a mast. Each illuminated color has the movement's glyph.
  const points = turn === "uturn" ? [
    [0.12, -0.36], [0.36, -0.36], [0.36, 0.16], [0.23, 0.38],
    [-0.15, 0.38], [-0.32, 0.20], [-0.32, -0.04], [-0.44, -0.04],
    [-0.20, -0.36], [0.04, -0.04], [-0.08, -0.04], [-0.08, 0.12],
    [-0.03, 0.16], [0.08, 0.16], [0.12, 0.12],
  ] : [
    [-0.12, -0.36], [0.12, -0.36], [0.12, 0.02],
    [0.36, 0.02], [0, 0.40], [-0.36, 0.02], [-0.12, 0.02],
  ];
  const shape = new THREE.Shape();
  points.forEach(([x, y], index) => index ? shape.lineTo(x, y) : shape.moveTo(x, y));
  shape.closePath();
  const geometry = new THREE.ShapeGeometry(shape);
  if (turn === "left") geometry.rotateZ(Math.PI / 2);
  if (turn === "right") geometry.rotateZ(-Math.PI / 2);
  return geometry;
}

function addSignalHead(head, headingRad, elevation = 0) {
  const group = new THREE.Group();
  group.position.set(head.position_xz[0], elevation, head.position_xz[1]);
  const frontX = -Math.cos(headingRad);
  const frontZ = -Math.sin(headingRad);
  group.rotation.y = Math.atan2(frontX, frontZ);
  const housing = new THREE.Mesh(
    new THREE.BoxGeometry(head.width_m || 1.05, 2.7, 0.36),
    new THREE.MeshStandardMaterial({ color: 0x111719, roughness: 0.55 }),
  );
  housing.position.set(0, 5.6, 0);
  group.add(housing);

  const bulbs = {};
  const specifications = [
    ["red", 0xff171f, 6.48],
    ["yellow", 0xffbf00, 5.6],
    ["green", 0x00ed45, 4.72],
  ];
  for (const [name, color, y] of specifications) {
    const material = new THREE.MeshStandardMaterial({
      color: 0x080a0b,
      emissive: color,
      emissiveIntensity: 0,
      roughness: 0.38,
      // Emissive signal colors must not wash out to white under the scene's
      // filmic tone mapping. Fog still applies; this is not a HUD element.
      toneMapped: false,
    });
    // A traffic-signal lens is directional.  A glowing sphere is visible
    // from the back and makes a perpendicular approach look as if it also
    // controls the ego lane.  FrontSide arrow geometry keeps the illuminated
    // aspect visible only to traffic approaching the face of this head.
    // Broader stems survive the cockpit image resolution without billboards,
    // zoom-dependent scaling or any semantic overlay.
    const bulb = new THREE.Mesh(signalArrowGeometry(head.turn), material);
    const glyphScale = Math.min(1, (head.width_m || 1.05) / 1.05);
    bulb.scale.setScalar(glyphScale);
    bulb.position.set(0, y, 0.205);
    group.add(bulb);
    bulbs[name] = bulb;
  }
  signalControllers.push({
    connectorId: head.connector_id,
    turn: head.turn,
    bulbs,
  });
  staticGroup.add(group);
}

function addSignal(signal) {
  const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x273036, roughness: 0.6 });
  const [poleX, poleZ] = signal.pole_position_xz;
  const [armX, armZ] = signal.arm_end_xz;
  const elevation = Number(signal.elevation_m || 0);

  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.13, 0.17, 5.4, 10),
    poleMaterial,
  );
  pole.position.set(poleX, elevation + 2.7, poleZ);
  pole.castShadow = true;
  staticGroup.add(pole);

  const dx = armX - poleX;
  const dz = armZ - poleZ;
  const armLength = Math.hypot(dx, dz);
  if (armLength > 0.05) {
    const arm = new THREE.Mesh(
      new THREE.BoxGeometry(armLength, 0.13, 0.13),
      poleMaterial,
    );
    arm.position.set(
      (poleX + armX) / 2,
      elevation + 5.35,
      (poleZ + armZ) / 2,
    );
    arm.rotation.y = -Math.atan2(dz, dx);
    arm.castShadow = true;
    staticGroup.add(arm);
  }

  for (const head of signal.heads || []) {
    addSignalHead(head, signal.heading_rad, elevation);
  }
}

function updateSignals(states = {}) {
  for (const controller of signalControllers) {
    const active = states[controller.connectorId] || "red";
    for (const [name, bulb] of Object.entries(controller.bulbs)) {
      const on = name === active;
      bulb.material.emissiveIntensity = on ? 1.8 : 0;
      if (on) bulb.material.color.setHex(0x000000); // Emission only, no sun tint.
      else bulb.material.color.setHex(0x080a0b);
    }
  }
}

function createTaperedCabinGeometry(width, height, length) {
  const bottomX = width / 2;
  const topX = width * 0.39;
  const frontBottom = length / 2;
  const rearBottom = -length / 2;
  const frontTop = length * 0.23;
  const rearTop = -length * 0.36;
  const positions = new Float32Array([
    -bottomX, 0, frontBottom, bottomX, 0, frontBottom,
    bottomX, 0, rearBottom, -bottomX, 0, rearBottom,
    -topX, height, frontTop, topX, height, frontTop,
    topX, height, rearTop, -topX, height, rearTop,
  ]);
  const indices = [
    0, 1, 5, 0, 5, 4,
    1, 2, 6, 1, 6, 5,
    2, 3, 7, 2, 7, 6,
    3, 0, 4, 3, 4, 7,
    4, 5, 6, 4, 6, 7,
    0, 3, 2, 0, 2, 1,
  ];
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function createVehicle(actor) {
  const [width, height, length] = actor.dimensions_m;
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: actor.color || (actor.is_focus ? "#31b6e7" : "#f59e42"),
    roughness: 0.3,
    metalness: 0.42,
  });
  const trimMaterial = new THREE.MeshStandardMaterial({
    color: 0x11171b, roughness: 0.72, metalness: 0.18,
  });
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0x152a35, roughness: 0.12, metalness: 0.58,
  });
  const wheelMaterial = new THREE.MeshStandardMaterial({
    color: 0x0c0e10, roughness: 0.96,
  });
  const rimMaterial = new THREE.MeshStandardMaterial({
    color: 0x8f989d, roughness: 0.32, metalness: 0.78,
  });
  bodyMaterial.userData.bevRole = "body";
  for (const mat of [trimMaterial, glassMaterial, wheelMaterial]) mat.userData.bevRole = "dark";

  const bodyHeight = height * 0.43;
  const body = new THREE.Mesh(
    new RoundedBoxGeometry(width, bodyHeight, length, 5, 0.15),
    bodyMaterial,
  );
  body.position.y = 0.25 + bodyHeight / 2;
  body.castShadow = true;
  group.add(body);

  const hood = new THREE.Mesh(
    new RoundedBoxGeometry(width * 0.9, height * 0.16, length * 0.29, 4, 0.09),
    bodyMaterial,
  );
  hood.position.set(0, height * 0.58, length * 0.335);
  hood.castShadow = true;
  group.add(hood);

  const cabinWidth = width * 0.79;
  const cabinHeight = height * 0.47;
  const cabinLength = length * 0.49;
  const cabinBaseY = height * 0.48;
  const cabinCenterZ = -length * 0.055;
  const cabin = new THREE.Mesh(
    createTaperedCabinGeometry(cabinWidth, cabinHeight, cabinLength),
    bodyMaterial,
  );
  cabin.position.set(0, cabinBaseY, cabinCenterZ);
  cabin.castShadow = true;
  group.add(cabin);

  const windshieldAngle = Math.atan2(
    -cabinLength * 0.27, cabinHeight,
  );
  const windshield = new THREE.Mesh(
    new RoundedBoxGeometry(
      cabinWidth * 0.76, cabinHeight * 0.58, 0.035, 3, 0.035,
    ),
    glassMaterial,
  );
  windshield.rotation.x = windshieldAngle;
  windshield.position.set(
    0,
    cabinBaseY + cabinHeight * 0.52,
    cabinCenterZ + cabinLength * 0.365,
  );
  group.add(windshield);

  const rearWindowAngle = Math.atan2(
    cabinLength * 0.14, cabinHeight,
  );
  const rearWindow = new THREE.Mesh(
    new RoundedBoxGeometry(
      cabinWidth * 0.72, cabinHeight * 0.5, 0.035, 3, 0.035,
    ),
    glassMaterial,
  );
  rearWindow.rotation.x = rearWindowAngle;
  rearWindow.position.set(
    0,
    cabinBaseY + cabinHeight * 0.52,
    cabinCenterZ - cabinLength * 0.43,
  );
  group.add(rearWindow);

  for (const side of [-1, 1]) {
    const windowX = side * cabinWidth * 0.445;
    for (const [zOffset, panelLength] of [
      [cabinLength * 0.13, cabinLength * 0.32],
      [-cabinLength * 0.23, cabinLength * 0.3],
    ]) {
      const sideWindow = new THREE.Mesh(
        new RoundedBoxGeometry(
          0.03, cabinHeight * 0.5, panelLength, 3, 0.03,
        ),
        glassMaterial,
      );
      sideWindow.position.set(
        windowX, cabinBaseY + cabinHeight * 0.52,
        cabinCenterZ + zOffset,
      );
      group.add(sideWindow);
    }
    const mirrorCenterX = side * width * 0.46;
    const mirrorZ = cabinCenterZ + cabinLength * 0.28;
    const mirrorY = cabinBaseY + cabinHeight * 0.38;
    const mirror = new THREE.Mesh(
      new RoundedBoxGeometry(width * 0.13, 0.12, 0.22, 3, 0.04),
      bodyMaterial,
    );
    mirror.position.set(mirrorCenterX, mirrorY, mirrorZ);
    group.add(mirror);

    const mountX = side * cabinWidth * 0.43;
    const mirrorInnerX = mirrorCenterX - side * width * 0.065;
    const stemLength = Math.abs(mirrorInnerX - mountX);
    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.025, 0.035, stemLength, 8),
      trimMaterial,
    );
    stem.rotation.z = Math.PI / 2;
    stem.position.set(
      (mountX + mirrorInnerX) / 2,
      mirrorY - 0.025,
      mirrorZ,
    );
    group.add(stem);

    const mirrorBase = new THREE.Mesh(
      new RoundedBoxGeometry(0.055, 0.17, 0.19, 3, 0.025),
      trimMaterial,
    );
    mirrorBase.position.set(mountX, mirrorY - 0.015, mirrorZ);
    group.add(mirrorBase);

    const mirrorGlass = new THREE.Mesh(
      new RoundedBoxGeometry(width * 0.095, 0.075, 0.025, 3, 0.02),
      glassMaterial,
    );
    mirrorGlass.position.set(
      mirrorCenterX, mirrorY + 0.005, mirrorZ - 0.115,
    );
    group.add(mirrorGlass);
  }

  const roof = new THREE.Mesh(
    new RoundedBoxGeometry(
      cabinWidth * 0.79, 0.09, cabinLength * 0.61, 4, 0.045,
    ),
    bodyMaterial,
  );
  roof.position.set(
    0, cabinBaseY + cabinHeight + 0.015,
    cabinCenterZ - cabinLength * 0.065,
  );
  group.add(roof);

  const wheelRadius = Math.min(0.34, height * 0.22);
  for (const x of [-width * 0.515, width * 0.515]) {
    for (const z of [-length * 0.31, length * 0.31]) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(wheelRadius, wheelRadius, 0.24, 24),
        wheelMaterial,
      );
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, wheelRadius, z);
      wheel.castShadow = true;
      group.add(wheel);
      const rim = new THREE.Mesh(
        new THREE.CylinderGeometry(wheelRadius * 0.52, wheelRadius * 0.52, 0.018, 16),
        rimMaterial,
      );
      rim.rotation.z = Math.PI / 2;
      rim.position.set(x + Math.sign(x) * 0.13, wheelRadius, z);
      group.add(rim);
    }
  }

  for (const z of [-length / 2 - 0.045, length / 2 + 0.045]) {
    const bumper = new THREE.Mesh(
      new RoundedBoxGeometry(width * 0.82, 0.13, 0.09, 3, 0.035),
      trimMaterial,
    );
    bumper.position.set(0, 0.33, z);
    group.add(bumper);
  }
  const grille = new THREE.Mesh(
    new RoundedBoxGeometry(width * 0.48, 0.18, 0.035, 3, 0.025),
    trimMaterial,
  );
  grille.position.set(0, height * 0.34, length / 2 + 0.085);
  group.add(grille);

  const headlights = [];
  const tailLights = [];
  const indicators = { left: [], right: [] };
  for (const x of [-width * 0.31, width * 0.31]) {
    const lamp = new THREE.Mesh(
      new RoundedBoxGeometry(width * 0.17, 0.13, 0.045, 3, 0.035),
      new THREE.MeshStandardMaterial({
        color: 0x303536,
        emissive: 0xf2f7db,
        emissiveIntensity: 0.02,
      }),
    );
    lamp.position.set(x, height * 0.43, length / 2 + 0.09);
    group.add(lamp);
    headlights.push(lamp);

    const tail = new THREE.Mesh(
      new RoundedBoxGeometry(width * 0.15, 0.14, 0.045, 3, 0.035),
      new THREE.MeshStandardMaterial({
        color: 0x351315,
        emissive: 0xff2028,
        emissiveIntensity: 0.05,
      }),
    );
    tail.position.set(x, height * 0.43, -length / 2 - 0.09);
    group.add(tail);
    tailLights.push(tail);
  }
  for (const [side, x] of [["left", -width * 0.44], ["right", width * 0.44]]) {
    for (const z of [-length / 2 - 0.03, length / 2 + 0.03]) {
      const indicator = new THREE.Mesh(
        new RoundedBoxGeometry(0.12, 0.075, 0.045, 2, 0.025),
        new THREE.MeshStandardMaterial({
          color: 0x39240c,
          emissive: 0xffa316,
          emissiveIntensity: 0.03,
        }),
      );
      indicator.position.set(x, height * 0.42, z + Math.sign(z) * 0.065);
      group.add(indicator);
      indicators[side].push(indicator);
    }
  }
  group.userData.lights = { headlights, tailLights, indicators };
  return group;
}

function updateVehicleLights(object, states = {}, simTime = 0) {
  const lights = object.userData.lights;
  if (!lights) return;
  const headlightLevel = states.high_beam ? 3.8 : states.low_beam ? 1.7 : 0.02;
  for (const lamp of lights.headlights) {
    lamp.material.emissiveIntensity = headlightLevel;
    lamp.material.color.setHex(headlightLevel > 0.1 ? 0xeef4da : 0x303536);
  }
  const rearLevel = states.brake_light ? 3.5 : states.tail_light ? 0.8 : 0.05;
  for (const lamp of lights.tailLights) lamp.material.emissiveIntensity = rearLevel;
  const blinkOn = Math.floor(simTime * 2) % 2 === 0;
  const leftOn = blinkOn && (states.left_indicator || states.hazard);
  const rightOn = blinkOn && (states.right_indicator || states.hazard);
  for (const lamp of lights.indicators.left) {
    lamp.material.emissiveIntensity = leftOn ? 3.2 : 0.03;
  }
  for (const lamp of lights.indicators.right) {
    lamp.material.emissiveIntensity = rightOn ? 3.2 : 0.03;
  }
}

const pedestrianGeometry = {
  head: new THREE.SphereGeometry(0.16, 14, 10),
  hair: new THREE.SphereGeometry(0.163, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.5),
  nose: new THREE.SphereGeometry(0.023, 8, 6),
  hand: new THREE.SphereGeometry(0.048, 9, 7),
  eye: new THREE.SphereGeometry(0.011, 7, 5),
  neck: new THREE.CylinderGeometry(0.052, 0.058, 0.12, 9),
  torso: new THREE.CylinderGeometry(0.225, 0.165, 0.53, 9),
  pelvis: new RoundedBoxGeometry(0.32, 0.19, 0.20, 4, 0.065),
  limb: new THREE.CylinderGeometry(1, 1, 1, 8),
  shoe: new RoundedBoxGeometry(0.13, 0.10, 0.25, 3, 0.04),
  mouth: new RoundedBoxGeometry(0.055, 0.009, 0.009, 2, 0.004),
};
const pedestrianSkinMaterial = new THREE.MeshStandardMaterial({
  color: 0xe1ad87,
  roughness: 0.88,
});
const pedestrianHairMaterial = new THREE.MeshStandardMaterial({
  color: 0x30241f,
  roughness: 0.94,
});
const pedestrianShoeMaterial = new THREE.MeshStandardMaterial({
  color: 0x25292c,
  roughness: 0.82,
});
const pedestrianEyeMaterial = new THREE.MeshStandardMaterial({
  color: 0x17191a,
  roughness: 0.75,
});
const pedestrianMouthMaterial = new THREE.MeshStandardMaterial({
  color: 0x8a5148,
  roughness: 0.85,
});

function pedestrianMesh(geometry, material) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function pedestrianLimb(start, end, radius, material) {
  const direction = end.clone().sub(start);
  const mesh = pedestrianMesh(pedestrianGeometry.limb, material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.scale.set(radius, direction.length(), radius);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0), direction.normalize(),
  );
  return mesh;
}

function createPedestrian(actor) {
  const group = new THREE.Group();
  const model = new THREE.Group();
  const clothingColor = new THREE.Color(actor.color || "#f0b84b");
  const topMaterial = new THREE.MeshStandardMaterial({
    color: clothingColor,
    roughness: 0.84,
  });
  const trouserMaterial = new THREE.MeshStandardMaterial({
    color: clothingColor.clone().multiplyScalar(0.35),
    roughness: 0.9,
  });

  const torso = pedestrianMesh(pedestrianGeometry.torso, topMaterial);
  torso.position.y = 1.18;
  torso.scale.z = 0.67;
  model.add(torso);
  const pelvis = pedestrianMesh(pedestrianGeometry.pelvis, trouserMaterial);
  pelvis.position.y = 0.88;
  model.add(pelvis);

  const neck = pedestrianMesh(pedestrianGeometry.neck, pedestrianSkinMaterial);
  neck.position.y = 1.47;
  model.add(neck);

  const head = pedestrianMesh(pedestrianGeometry.head, pedestrianSkinMaterial);
  head.position.y = 1.61;
  head.scale.set(0.80, 1.0, 0.84);
  model.add(head);
  const hair = pedestrianMesh(pedestrianGeometry.hair, pedestrianHairMaterial);
  hair.position.y = 1.625;
  hair.scale.set(0.82, 1.02, 0.86);
  model.add(hair);
  // The face and shoes point along local +Z, making walking direction readable
  // without adding an artificial label or arrow to the LLM camera image.
  const nose = pedestrianMesh(pedestrianGeometry.nose, pedestrianSkinMaterial);
  nose.position.set(0, 1.60, 0.139);
  model.add(nose);
  for (const x of [-0.047, 0.047]) {
    const eye = pedestrianMesh(pedestrianGeometry.eye, pedestrianEyeMaterial);
    eye.position.set(x, 1.645, 0.132);
    model.add(eye);
  }
  const mouth = pedestrianMesh(pedestrianGeometry.mouth, pedestrianMouthMaterial);
  mouth.position.set(0, 1.555, 0.133);
  model.add(mouth);

  for (const sign of [-1, 1]) {
    const shoulder = new THREE.Vector3(sign * 0.22, 1.37, -sign * 0.02);
    const elbow = new THREE.Vector3(sign * 0.265, 1.14, -sign * 0.06);
    const wrist = new THREE.Vector3(sign * 0.235, 0.94, -sign * 0.045);
    model.add(pedestrianLimb(shoulder, elbow, 0.058, topMaterial));
    model.add(pedestrianLimb(elbow, wrist, 0.048, topMaterial));
    const hand = pedestrianMesh(pedestrianGeometry.hand, pedestrianSkinMaterial);
    hand.position.copy(wrist);
    model.add(hand);

    const hip = new THREE.Vector3(sign * 0.09, 0.89, sign * 0.025);
    const knee = new THREE.Vector3(sign * 0.105, 0.52, sign * 0.045);
    const ankle = new THREE.Vector3(sign * 0.105, 0.16, sign * 0.055);
    model.add(pedestrianLimb(hip, knee, 0.074, trouserMaterial));
    model.add(pedestrianLimb(knee, ankle, 0.063, trouserMaterial));
    const shoe = pedestrianMesh(pedestrianGeometry.shoe, pedestrianShoeMaterial);
    shoe.position.set(ankle.x, 0.085, ankle.z + 0.085);
    model.add(shoe);
  }

  const [width = 0.55, height = 1.72] = actor.dimensions_m || [];
  const horizontalScale = THREE.MathUtils.clamp(Number(width) / 0.55, 0.82, 1.22);
  model.scale.set(horizontalScale, Number(height) / 1.72, horizontalScale);
  group.add(model);
  return group;
}

function preparePath(points) {
  const vectors = points.map(([x, z]) => new THREE.Vector3(x, 0, z));
  const cumulative = [0];
  for (let index = 1; index < vectors.length; index += 1) {
    cumulative.push(cumulative.at(-1) + vectors[index].distanceTo(vectors[index - 1]));
  }
  return { vectors, cumulative, length: cumulative.at(-1) || 0.001 };
}

function samplePath(path, distance) {
  const wrapped = ((distance % path.length) + path.length) % path.length;
  let index = 1;
  while (index < path.cumulative.length && path.cumulative[index] < wrapped) index += 1;
  index = Math.min(index, path.vectors.length - 1);
  const start = path.vectors[index - 1];
  const end = path.vectors[index];
  const segmentStart = path.cumulative[index - 1];
  const segmentLength = Math.max(0.001, path.cumulative[index] - segmentStart);
  const ratio = (wrapped - segmentStart) / segmentLength;
  const position = start.clone().lerp(end, ratio);
  const direction = end.clone().sub(start).normalize();
  return { position, direction };
}

const actors = new Map();
let egoActor = null;
let sceneCenterWorldXY = [0, 0];

function actorPosition(actor) {
  if (actor.position_world_xy) {
    // Vehicle positions from the live encoder are already body centres.
    // Do not apply a second SUMO front-bumper offset in the browser.
    return new THREE.Vector3(
      actor.position_world_xy[0] - sceneCenterWorldXY[0],
      Number(actor.elevation_m || 0),
      sceneCenterWorldXY[1] - actor.position_world_xy[1],
    );
  }
  if (!actor.path_xz || actor.path_xz.length < 2) return null;
  return samplePath(preparePath(actor.path_xz), actor.initial_s_m || 0).position;
}

function actorRotation(actor, previous) {
  if (Number.isFinite(actor.yaw_rad)) return Math.PI / 2 + actor.yaw_rad;
  if (actor.path_xz && actor.path_xz.length >= 2) {
    const sampled = samplePath(
      preparePath(actor.path_xz), actor.initial_s_m || 0,
    );
    return Math.atan2(sampled.direction.x, sampled.direction.z);
  }
  const target = actorPosition(actor);
  if (previous && target && target.distanceTo(previous) > 0.005) {
    const direction = target.clone().sub(previous).normalize();
    return Math.atan2(direction.x, direction.z);
  }
  return 0;
}

function addActor(actor, now = performance.now(), durationMs = 0) {
  const position = actorPosition(actor);
  if (!position) return null;
  const object = actor.kind === "vehicle" ? createVehicle(actor) : createPedestrian(actor);
  const rotation = actorRotation(actor, null);
  object.position.copy(position);
  object.rotation.y = rotation;
  const runtime = {
    ...actor,
    object,
    fromPosition: position.clone(),
    targetPosition: position.clone(),
    fromRotation: rotation,
    targetRotation: rotation,
    updateStartedAt: now,
    updateDurationMs: durationMs,
    speedMps: Number(actor.speed_mps || 0),
    signalState: actor.signals || {},
    direction: new THREE.Vector3(Math.sin(rotation), 0, Math.cos(rotation)),
  };
  actors.set(actor.id, runtime);
  actorGroup.add(object);
  if (actor.is_focus || actor.id === "ego") egoActor = runtime;
  if (actor.kind === "vehicle") {
    updateVehicleLights(object, runtime.signalState, simulationTime);
  }
  return runtime;
}

function shortestAngle(from, to) {
  let delta = (to - from) % (Math.PI * 2);
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return from + delta;
}

function upsertActor(actor, now, durationMs) {
  let runtime = actors.get(actor.id);
  if (!runtime || runtime.kind !== actor.kind) {
    if (runtime) actorGroup.remove(runtime.object);
    return addActor(actor, now, durationMs);
  }
  const target = actorPosition(actor);
  if (!target) return runtime;
  const previousTarget = runtime.targetPosition.clone();
  runtime.fromPosition.copy(runtime.object.position);
  runtime.targetPosition.copy(target);
  runtime.fromRotation = runtime.object.rotation.y;
  runtime.targetRotation = shortestAngle(
    runtime.fromRotation,
    actorRotation(actor, previousTarget),
  );
  runtime.updateStartedAt = now;
  runtime.updateDurationMs = durationMs;
  runtime.speedMps = Number(actor.speed_mps || 0);
  runtime.control = actor.control || runtime.control;
  runtime.signalState = actor.signals || {};
  runtime.crashed = Boolean(actor.crashed);
  if (actor.is_focus) egoActor = runtime;
  if (runtime.kind === "vehicle") {
    updateVehicleLights(runtime.object, runtime.signalState, simulationTime);
  }
  return runtime;
}

function updateActors(now) {
  for (const actor of actors.values()) {
    const ratio = actor.updateDurationMs <= 0
      ? 1
      : Math.min(1, Math.max(0, (now - actor.updateStartedAt) / actor.updateDurationMs));
    actor.object.position.lerpVectors(actor.fromPosition, actor.targetPosition, ratio);
    actor.object.rotation.y = THREE.MathUtils.lerp(
      actor.fromRotation, actor.targetRotation, ratio,
    );
    actor.direction.set(
      Math.sin(actor.object.rotation.y), 0, Math.cos(actor.object.rotation.y),
    );
    if (actor.kind === "vehicle") {
      updateVehicleLights(actor.object, actor.signalState, simulationTime);
    }
  }
}

let cameraMode = "cockpit";
let simulationTime = 0;
const cameraYawAxis = new THREE.Vector3(0, 1, 0);
const renderViewSize = new THREE.Vector2();

function placeCockpitCamera(yawOffsetRad = 0, aspect = null, fovDeg = 68) {
  if (!egoActor) return;
  const position = egoActor.object.position;
  const direction = egoActor.direction.clone();
  if (yawOffsetRad) direction.applyAxisAngle(cameraYawAxis, yawOffsetRad);
  camera.fov = fovDeg;
  camera.aspect = aspect || window.innerWidth / window.innerHeight;
  // The vehicle actor origin is its body centre, not SUMO's front bumper.
  // Keep every optical view at the same driver's-eye origin. Only gaze
  // direction changes, so an actor cannot appear/disappear because the
  // side-view camera was translated outside the vehicle.
  camera.position.copy(position).addScaledVector(egoActor.direction, 0.72);
  camera.position.y = position.y + 1.62;
  const target = camera.position.clone().addScaledVector(direction, 42);
  target.y = position.y + 1.34;
  camera.lookAt(target);
  camera.updateProjectionMatrix();
}

function updateCamera() {
  if (!egoActor || cameraMode === "overview") return;
  const position = egoActor.object.position;
  const direction = egoActor.direction;
  if (cameraMode === "cockpit") {
    placeCockpitCamera();
  } else {
    camera.fov = 58;
    camera.aspect = window.innerWidth / window.innerHeight;
    const desired = position.clone().addScaledVector(direction, -12);
    desired.y = position.y + 7.2;
    camera.position.lerp(desired, 0.12);
    const target = position.clone().addScaledVector(direction, 12);
    target.y = position.y + 1.0;
    camera.lookAt(target);
  }
  camera.updateProjectionMatrix();
}

function renderDriverObservation() {
  renderer.getSize(renderViewSize);
  const width = Math.round(renderViewSize.x);
  const height = Math.round(renderViewSize.y);
  const frontHeight = Math.min(height, Math.round(width * 3 / 4));
  const sideBandHeight = height - frontHeight;
  const margin = Math.max(6, Math.round(sideBandHeight * 0.04));
  const border = Math.max(3, Math.round(width * 0.003));
  const paneWidth = Math.max(1, Math.floor(width / 2) - margin * 2);
  const paneHeight = Math.max(1, sideBandHeight - margin * 2);
  const paneY = frontHeight + margin;
  const oldClear = renderer.getClearColor(new THREE.Color()).clone();
  const oldAlpha = renderer.getClearAlpha();

  // Reserve a compact band above the windscreen instead of floating panes
  // over it. The lower camera remains exactly 4:3, so existing signal and
  // road visibility is unchanged.
  document.documentElement.style.setProperty(
    "--driver-main-top", `${sideBandHeight}px`,
  );
  document.documentElement.style.setProperty(
    "--driver-main-height", `${frontHeight}px`,
  );
  renderer.setScissorTest(false);
  renderer.setClearColor(0x071017, 1);
  renderer.clear(true, true, true);
  renderer.setScissorTest(true);
  renderer.setViewport(0, 0, width, frontHeight);
  renderer.setScissor(0, 0, width, frontHeight);
  placeCockpitCamera(0, width / frontHeight);
  renderer.render(scene, camera);

  // The side panes are simultaneous driver glances rather than mirrors.
  // Physical left occupies the upper-left half and physical right the
  // upper-right half, both outside the forward windscreen image.
  const panes = [
    { x: margin, yaw: THREE.MathUtils.degToRad(75) },
    { x: Math.floor(width / 2) + margin,
      yaw: -THREE.MathUtils.degToRad(75) },
  ];
  for (const pane of sideBandHeight >= 96 ? panes : []) {
    renderer.setViewport(
      pane.x - border, paneY - border,
      paneWidth + border * 2, paneHeight + border * 2,
    );
    renderer.setScissor(
      pane.x - border, paneY - border,
      paneWidth + border * 2, paneHeight + border * 2,
    );
    renderer.setClearColor(0x071017, 1);
    renderer.clear(true, true, true);

    renderer.setViewport(pane.x, paneY, paneWidth, paneHeight);
    renderer.setScissor(pane.x, paneY, paneWidth, paneHeight);
    placeCockpitCamera(pane.yaw, paneWidth / paneHeight, 44);
    renderer.render(scene, camera);
  }
  renderer.setClearColor(oldClear, oldAlpha);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, frontHeight);
  placeCockpitCamera(0, width / frontHeight);
}

function setCameraMode(mode) {
  cameraMode = mode;
  controls.enabled = mode === "overview";
  cockpitOverlay.classList.toggle("hidden", mode !== "cockpit");
  document.querySelectorAll("[data-camera]").forEach((button) => {
    button.classList.toggle("active", button.dataset.camera === mode);
  });
  if (mode === "overview") {
    camera.fov = 52;
    camera.position.set(88, 118, 96);
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
  }
  applyWeatherViewTuning();
}

function clearGroup(group) {
  while (group.children.length) group.remove(group.children[0]);
}

function buildScene(data, { live = false } = {}) {
  lidarBEVRenderer?.reset();
  clearGroup(staticGroup);
  clearGroup(actorGroup);
  signalControllers.length = 0;
  actors.clear();
  egoActor = null;
  sceneCenterWorldXY = data.map.center_world_xy.map(Number);
  mapLabel.textContent = `${data.map.id} · junction ${data.map.junction_id}`;
  for (const road of data.static.roads) {
    staticGroup.add(polygonMesh(
      road.polygon_xz,
      roadMaterial,
      0.015 + (road.elevation_m || 0),
    ));
  }
  for (const surface of data.static.junction_surfaces) {
    if (surface.polygon_xz.length >= 3) {
      staticGroup.add(polygonMesh(
        surface.polygon_xz,
        junctionMaterial,
        0.02 + (surface.elevation_m || 0),
      ));
    }
  }
  data.static.markings.forEach(addPolyline);
  (data.static.ground_arrows || []).forEach(addGroundArrow);
  for (const stopLine of data.static.stop_lines) {
    addThickSegment(stopLine.points_xz, 0xf2efe3, 0.55,
      0.075 + (stopLine.elevation_m || 0));
  }
  for (const crosswalk of data.static.crosswalks) {
    for (const stripe of crosswalk.stripes_xz) {
      staticGroup.add(polygonMesh(stripe, crosswalkMaterial, 0.085));
    }
  }
  data.static.buildings.forEach(addBuilding);
  data.static.signals.forEach(addSignal);
  if (!live) data.snapshot.actors.forEach((actor) => addActor(actor));
  simulationTime = Number(data.snapshot.sim_time_s || 0);
  updateActors(performance.now());
  updateSignals({});
  updateEnvironment(data.snapshot || {});
  controlSourceLabel.textContent = egoActor?.control === "llm"
    ? "LLM 预览"
    : egoActor ? "SUMO 预览" : "—";
  setCameraMode(query.get("view") || "cockpit");
  loading.hidden = true;
}

function updateEnvironment(environment = {}) {
  const rawDaylight = environment.daylight_level
    ?? (environment.is_night ? 16 : 100);
  const daylight = Math.max(
    0, Math.min(100, Number(rawDaylight)),
  ) / 100;
  const weather = environment.weather || "sunny";
  const profile = WEATHER_PROFILES[weather] || WEATHER_PROFILES.sunny;
  const level = Math.max(0.14, daylight * profile.darkening);
  const horizon = new THREE.Color(0xc3d2d6).multiplyScalar(level);
  const zenith = new THREE.Color(0x789fb3).multiplyScalar(level);
  const weatherHaze = Math.min(0.72, Number(profile.fog || 0) * 48);
  zenith.lerp(horizon, weatherHaze);
  paintSky(zenith, horizon);
  scene.fog.color.copy(horizon);
  sun.intensity = 0.5 + 2.5 * level;
  renderer.toneMappingExposure = 0.88 + 0.17 * level;

  const wet = Number(profile.wet || 0);
  const snowCover = Number(profile.snowCover || 0);
  const roadColor = new THREE.Color(0x303437);
  const junctionColor = new THREE.Color(0x373b3e);
  if (wet > 0) {
    roadColor.lerp(new THREE.Color(0x141b20), wet);
    junctionColor.lerp(new THREE.Color(0x192126), wet);
  }
  if (snowCover > 0) {
    roadColor.lerp(new THREE.Color(0x788083), snowCover * 0.56);
    junctionColor.lerp(new THREE.Color(0x7d8587), snowCover * 0.56);
  }
  roadMaterial.color.copy(roadColor);
  junctionMaterial.color.copy(junctionColor);
  roadMaterial.roughness = wet > 0 ? 0.92 - wet * 0.62 : 0.92;
  junctionMaterial.roughness = wet > 0 ? 0.9 - wet * 0.58 : 0.92;
  roadMaterial.metalness = 0.02 + wet * 0.12;
  junctionMaterial.metalness = 0.02 + wet * 0.1;
  groundMaterial.color.setHex(snowCover > 0 ? 0xdce4e2 : 0xb4bbaa);

  activeWeather = {
    name: weather,
    rain: Number(profile.rain || 0),
    snow: Number(profile.snow || 0),
    windSpeedMps: Number(environment.wind_speed_mps || 0),
    fogDensity: Number(profile.fog || WEATHER_PROFILES.sunny.fog),
  };
  document.body.dataset.weather = weather;
  document.body.dataset.rainIntensity = String(activeWeather.rain);
  document.body.dataset.snowIntensity = String(activeWeather.snow);
  applyWeatherViewTuning();
}

function wrapped(value, span) {
  return ((value % span) + span) % span;
}

function weatherViewTuning() {
  if (cameraMode === "overview") {
    return { particle: 0.18, streak: 0.28, opacity: 0.52, fog: 0.35, spread: 2.1 };
  }
  if (cameraMode === "follow") {
    return { particle: 0.55, streak: 0.68, opacity: 0.78, fog: 0.72, spread: 1.25 };
  }
  return { particle: 1, streak: 1, opacity: 1, fog: 1, spread: 1 };
}

function applyWeatherViewTuning() {
  const tuning = weatherViewTuning();
  scene.fog.density = activeWeather.fogDensity * tuning.fog;
  updateWeatherParticles(simulationTime);
}

function updateWeatherParticles(timeS) {
  const tuning = weatherViewTuning();
  const rainCount = Math.round(
    rainSystem.userData.seeds.length * activeWeather.rain * tuning.particle,
  );
  rainSystem.visible = rainCount > 0;
  rainSystem.geometry.setDrawRange(0, rainCount * 2);
  if (rainCount > 0) {
    const positions = rainSystem.geometry.attributes.position.array;
    const windSlant = Math.max(
      -0.75, Math.min(0.75, activeWeather.windSpeedMps * 0.045),
    );
    for (let index = 0; index < rainCount; index += 1) {
      const seed = rainSystem.userData.seeds[index];
      const x = (seed.x - 0.5) * 42 * tuning.spread;
      const y = -6 + wrapped(seed.y * 22 - timeS * 17.5, 22);
      const z = -2.5 - seed.z * 52 * tuning.spread;
      const offset = index * 6;
      positions[offset] = x;
      positions[offset + 1] = y;
      positions[offset + 2] = z;
      positions[offset + 3] = x - windSlant;
      positions[offset + 4] = y
        + (1.05 + activeWeather.rain * 0.65) * tuning.streak;
      positions[offset + 5] = z - 0.18 * tuning.streak;
    }
    rainSystem.geometry.attributes.position.needsUpdate = true;
    rainSystem.material.opacity = (
      0.28 + activeWeather.rain * 0.3
    ) * tuning.opacity;
  }

  const snowCount = Math.round(
    snowSystem.userData.seeds.length * activeWeather.snow * tuning.particle,
  );
  snowSystem.visible = snowCount > 0;
  snowSystem.geometry.setDrawRange(0, snowCount);
  if (snowCount > 0) {
    const positions = snowSystem.geometry.attributes.position.array;
    for (let index = 0; index < snowCount; index += 1) {
      const seed = snowSystem.userData.seeds[index];
      const drift = Math.sin(timeS * 0.7 + seed.drift) * 0.85
        + activeWeather.windSpeedMps * timeS * 0.025;
      const offset = index * 3;
      positions[offset] = (seed.x - 0.5) * 42 * tuning.spread + drift;
      positions[offset + 1] = -6
        + wrapped(seed.y * 21 - timeS * 1.65, 21);
      positions[offset + 2] = -2.5 - seed.z * 52 * tuning.spread;
    }
    snowSystem.geometry.attributes.position.needsUpdate = true;
    snowSystem.material.opacity = (
      0.58 + activeWeather.snow * 0.28
    ) * tuning.opacity;
    snowSystem.material.size = 0.16 + activeWeather.snow * 0.12;
  }
}

let activeSceneKey = "";
let pendingLiveFrame = null;
let sceneRequest = null;
let lastFrameSequence = -1;
let streamEnded = false;

function liveSceneKey(spec) {
  return [
    spec.map_id,
    spec.revision,
    spec.center_world_xy?.[0],
    spec.center_world_xy?.[1],
    spec.radius_m,
  ].join(":");
}

function requestLiveScene(frame) {
  pendingLiveFrame = frame;
  const spec = frame.scene || {};
  const key = liveSceneKey(spec);
  if (key === activeSceneKey) return Promise.resolve();
  if (sceneRequest) return sceneRequest;
  const params = new URLSearchParams({
    map_id: spec.map_id,
    radius_m: String(spec.radius_m),
    center_x: String(spec.center_world_xy[0]),
    center_y: String(spec.center_world_xy[1]),
    demo: "0",
  });
  sceneRequest = fetch(`/api/scene?${params.toString()}`)
    .then((response) => {
      if (!response.ok) throw new Error(`scene API returned ${response.status}`);
      return response.json();
    })
    .then((data) => {
      buildScene(data, { live: true });
      activeSceneKey = key;
      sceneRequest = null;
      const newest = pendingLiveFrame;
      pendingLiveFrame = null;
      if (newest) applyLiveFrame(newest);
    })
    .catch((error) => {
      sceneRequest = null;
      loading.hidden = true;
      errorBox.hidden = false;
      errorBox.textContent = `无法加载实时地图：${error.message}`;
    });
  return sceneRequest;
}

function applyLiveFrame(frame) {
  if (frame.schema !== "vehiclearena-web3d-frame-v0.1") return;
  if (Number(frame.sequence) < lastFrameSequence) return;
  if (frame.stream_status === "ended") {
    streamEnded = true;
    connectionLabel.textContent = "仿真结束";
  }
  const key = liveSceneKey(frame.scene || {});
  if (key !== activeSceneKey) {
    requestLiveScene(frame);
    return;
  }
  const priorTime = simulationTime;
  simulationTime = Number(frame.sim_time_s || 0);
  lastFrameSequence = Number(frame.sequence);
  const durationMs = Math.max(
    0,
    Math.min(250, (simulationTime - priorTime) * 1000),
  );
  const now = performance.now();
  const visible = new Set();
  for (const actor of frame.actors || []) {
    visible.add(actor.id);
    upsertActor(actor, now, durationMs);
  }
  for (const [actorId, actor] of actors.entries()) {
    if (visible.has(actorId)) continue;
    actorGroup.remove(actor.object);
    actors.delete(actorId);
    if (actor === egoActor) egoActor = null;
  }
  // Explicit frame ownership wins over legacy IDs and actor iteration order.
  if (frame.focus_entity_id) egoActor = actors.get(frame.focus_entity_id) || null;
  updateSignals(frame.signals || {});
  updateEnvironment(frame.environment || {});
  streamEnded = frame.stream_status === "ended";
  connectionLabel.textContent = streamEnded ? "仿真结束" : "SUMO 实时";
  controlSourceLabel.textContent = egoActor?.control === "llm"
    ? "LLM → SUMO"
    : egoActor ? "SUMO NPC" : "—";
  simTimeLabel.textContent = `${simulationTime.toFixed(1)} s`;
}

document.querySelectorAll("[data-camera]").forEach((button) => {
  button.addEventListener("click", () => setCameraMode(button.dataset.camera));
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let continuousAnimationEnabled = false;

function animate() {
  if (!continuousAnimationEnabled) return;
  requestAnimationFrame(animate);
  updateActors(performance.now());
  updateCamera();
  updateWeatherParticles(simulationTime);
  if (controls.enabled) controls.update();
  const egoSpeed = egoActor ? egoActor.speedMps * 3.6 : 0;
  simTimeLabel.textContent = `${simulationTime.toFixed(1)} s`;
  speedLabel.textContent = `${Math.round(egoSpeed)} km/h`;
  if (captureMode && cameraMode === "cockpit" && egoActor) {
    renderDriverObservation();
  } else {
    renderer.render(scene, camera);
  }
}

const query = new URLSearchParams(window.location.search);
const sessionId = query.get("session_id");
const captureMode = query.get("capture") === "1";
if (captureMode) document.body.classList.add("capture-mode");
let lidarBEVRenderer = null;

window.vehicleArenaCaptureFrame = async (frame) => {
  if (!captureMode) throw new Error("capture mode is not enabled");
  const key = liveSceneKey(frame.scene || {});
  if (key !== activeSceneKey) await requestLiveScene(frame);
  applyLiveFrame(frame);
  // Normal viewing interpolates between physical frames. An LLM observation
  // must instead show the exact frozen boundary supplied by the engine.
  updateActors(performance.now() + 1000);
  updateCamera();
  updateWeatherParticles(simulationTime);
  renderDriverObservation();
  loading.hidden = true;
  errorBox.hidden = true;
  return {
    sim_time_s: simulationTime,
    frame_sequence: lastFrameSequence,
    focus_entity_id: egoActor?.id,
    weather: activeWeather.name,
    rain_intensity: activeWeather.rain,
    snow_intensity: activeWeather.snow,
    observation_layout: "front_with_upper_side_glance_strip",
    render_mode: continuousAnimationEnabled ? "continuous" : "on_demand",
  };
};

window.vehicleArenaCaptureLidarBEV = ({ sim_time_s, focus_entity_id, sensor }) => {
  if (!captureMode || continuousAnimationEnabled) throw new Error("LiDAR BEV requires on-demand capture mode");
  if (Math.abs(Number(sim_time_s) - simulationTime) > 1e-6 || egoActor?.id !== focus_entity_id) {
    throw new Error("LiDAR BEV frozen frame/focus mismatch");
  }
  lidarBEVRenderer ||= new LidarBEVRenderer(scene, renderer, {
    ground: groundMaterial, road: roadMaterial, junction: junctionMaterial,
    crosswalk: crosswalkMaterial, arrow: groundArrowMaterial,
  }, weatherGroup);
  return { ...lidarBEVRenderer.render(egoActor, actors, sensor),
    sim_time_s: simulationTime, frame_sequence: lastFrameSequence, render_mode: "on_demand" };
};

function connectLive() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}/api/live/ws?session_id=${encodeURIComponent(sessionId)}`,
  );
  connectionLabel.textContent = "正在连接";
  socket.addEventListener("open", () => {
    connectionLabel.textContent = "等待 SUMO";
  });
  socket.addEventListener("message", (event) => {
    try {
      applyLiveFrame(JSON.parse(event.data));
    } catch (error) {
      errorBox.hidden = false;
      errorBox.textContent = `实时帧无效：${error.message}`;
    }
  });
  socket.addEventListener("close", () => {
    if (streamEnded) return;
    connectionLabel.textContent = "连接中断，正在重连";
    window.setTimeout(connectLive, 1000);
  });
  socket.addEventListener("error", () => {
    connectionLabel.textContent = "连接异常";
  });
}

if (captureMode) {
  loading.textContent = "等待 VehicleArena 冻结帧…";
} else if (sessionId) {
  loading.textContent = "等待 VehicleArena/SUMO 实时帧…";
  connectLive();
} else {
  const sceneQuery = new URLSearchParams();
  for (const key of ["map_id", "junction_id", "radius_m"]) {
    if (query.has(key)) sceneQuery.set(key, query.get(key));
  }
  fetch(`/api/scene?${sceneQuery.toString()}`)
    .then((response) => {
      if (!response.ok) throw new Error(`scene API returned ${response.status}`);
      return response.json();
    })
    .then((data) => buildScene(data, { live: false }))
    .catch((error) => {
      loading.hidden = true;
      errorBox.hidden = false;
      errorBox.textContent = `无法加载三维场景：${error.message}`;
      console.error(error);
    });
}

// The interactive viewer needs a continuous animation loop.  Capture pages
// instead render exactly once inside vehicleArenaCaptureFrame; leaving a RAF
// loop running while the LLM API responds kept Chromium's software WebGL
// process busy for the entire episode.
if (!captureMode) {
  continuousAnimationEnabled = true;
  animate();
}
