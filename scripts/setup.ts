#!/usr/bin/env bun
import { mkdirSync, existsSync, symlinkSync } from "fs";
import { join } from "path";
import { $ } from "bun";

const DEFAULT_VERSION = "1.26.0";
const version = process.argv[2] ?? DEFAULT_VERSION;

const ROOT = join(import.meta.dir, "..");
const OUT_DIR = join(ROOT, "src", "infer", "lib", "onnxruntime");

function getAssetName(): { name: string; ext: "tgz" | "zip" } {
  const { platform, arch } = process;

  if (platform === "linux") {
    const onnxArch = arch === "arm64" ? "aarch64" : "x64";
    return { name: `onnxruntime-linux-${onnxArch}-${version}`, ext: "tgz" };
  }
  if (platform === "darwin") {
    const onnxArch = arch === "x64" ? "x86_64" : "arm64";
    return { name: `onnxruntime-osx-${onnxArch}-${version}`, ext: "tgz" };
  }
  if (platform === "win32") {
    const onnxArch = arch === "arm64" ? "arm64" : "x64";
    return { name: `onnxruntime-win-${onnxArch}-${version}`, ext: "zip" };
  }
  throw new Error(`Unsupported platform: ${platform} / ${arch}`);
}

async function download(url: string, dest: string): Promise<void> {
  console.log(`Downloading ${url}`);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
  await Bun.write(dest, resp);
}

async function extract(archive: string, destDir: string, ext: "tgz" | "zip"): Promise<void> {
  mkdirSync(destDir, { recursive: true });
  if (ext === "tgz") {
    await $`tar -xzf ${archive} -C ${destDir} --strip-components=1`;
  } else {
    const tmp = archive.replace(/\.zip$/, "_unzip_tmp");
    await $`unzip -q ${archive} -d ${tmp}`;
    // zip contains a single top-level dir — move its contents
    const [inner] = await $`ls ${tmp}`.text().then((t) => t.trim().split("\n"));
    await $`cp -r ${join(tmp, inner)}/. ${destDir}`;
    await $`rm -rf ${tmp}`;
  }
}

async function main() {
  const { name, ext } = getAssetName();
  const url = `https://github.com/microsoft/onnxruntime/releases/download/v${version}/${name}.${ext}`;
  const archive = join(OUT_DIR, `..`, `${name}.${ext}`);

  if (existsSync(join(OUT_DIR, "include"))) {
    console.log(`ONNX Runtime ${version} already present at ${OUT_DIR}`);
    return;
  }

  mkdirSync(join(OUT_DIR, ".."), { recursive: true });

  await download(url, archive);
  console.log(`Extracting to ${OUT_DIR}`);
  await extract(archive, OUT_DIR, ext);
  await $`rm -f ${archive}`;

  // onnxruntime cmake targets reference include/onnxruntime/ but tarball puts
  // headers directly in include/ — create a self-referencing symlink
  const includeOrt = join(OUT_DIR, "include", "onnxruntime");
  if (!existsSync(includeOrt)) {
    symlinkSync(".", includeOrt);
  }

  // onnxruntime Linux cmake targets reference lib64/ but tarball ships lib/
  const lib64 = join(OUT_DIR, "lib64");
  const lib = join(OUT_DIR, "lib");
  if (!existsSync(lib64) && existsSync(lib)) {
    symlinkSync("lib", lib64);
  }

  console.log(`Done: ${OUT_DIR}`);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
