import { existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";

interface Asset {
  url: string;
  dest: string;
}

const BASE_DIR = dirname(import.meta.path);

const assets: Asset[] = [
  {
    url: "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-male.mp4",
    dest: "video/head-pose-face-detection-male.mp4",
  },
];

async function downloadAsset(asset: Asset): Promise<void> {
  const destPath = join(BASE_DIR, asset.dest);
  if (existsSync(destPath)) {
    console.log(`[skip] ${asset.dest} (already exists)`);
    return;
  }

  const destDir = dirname(destPath);
  mkdirSync(destDir, { recursive: true });

  console.log(`[download] ${asset.dest} ...`);
  const response = await fetch(asset.url);
  if (!response.ok) {
    throw new Error(
      `Failed to download ${asset.url}: ${response.status} ${response.statusText}`
    );
  }

  await Bun.write(destPath, response);
  console.log(`[done] ${asset.dest}`);
}

async function main() {
  console.log(`Downloading ${assets.length} asset(s) to ${BASE_DIR}\n`);

  const results = await Promise.allSettled(assets.map(downloadAsset));

  const failures = results.filter((r) => r.status === "rejected");
  if (failures.length > 0) {
    console.error(`\n${failures.length} download(s) failed:`);
    for (const f of failures) {
      console.error(`  - ${(f as PromiseRejectedResult).reason}`);
    }
    process.exit(1);
  }

  console.log("\nAll assets downloaded.");
}

main();
