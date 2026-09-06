// The bun process mirrors its log to a rotating file, so a run that was not
// started from a terminal still leaves something to read. The location follows
// the platform convention, with a per-user fallback for when the machine-wide
// directory is not writable:
//
//   Windows  %PROGRAMDATA%\ptensor\  ->  %LOCALAPPDATA%\ptensor\logs\
//   macOS    /Library/Logs/ptensor/  ->  ~/Library/Logs/ptensor/
//   Linux    /var/log/ptensor/       ->  ~/.local/state/ptensor/
//
// `PTENSOR_VIEW_LOG_FILE` overrides the whole search with one explicit path.

import { mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { getRotatingFileSink } from '@logtape/file';
import type { LogSink } from '../shared/logging';

const FILE_NAME = 'ptensor-desktop.log';

/** Rotated once the file passes this size, keeping `MAX_FILES` generations. */
const MAX_SIZE = 5 * 1024 * 1024;
const MAX_FILES = 5;

export interface LogFile {
    sink: LogSink;
    path: string;
}

/** Paths to try, most preferred first. */
function candidates(): string[] {
    const override = process.env.PTENSOR_VIEW_LOG_FILE;
    if (override !== undefined && override.length > 0) {
        return [override];
    }
    const home = homedir();
    if (process.platform === 'win32') {
        const programData = process.env.PROGRAMDATA ?? 'C:\\ProgramData';
        const localAppData = process.env.LOCALAPPDATA ?? join(home, 'AppData', 'Local');
        return [
            join(programData, 'ptensor', FILE_NAME),
            join(localAppData, 'ptensor', 'logs', FILE_NAME),
        ];
    }
    if (process.platform === 'darwin') {
        return [
            join('/Library/Logs', 'ptensor', FILE_NAME),
            join(home, 'Library', 'Logs', 'ptensor', FILE_NAME),
        ];
    }
    return [
        join('/var/log', 'ptensor', FILE_NAME),
        join(home, '.local', 'state', 'ptensor', FILE_NAME),
    ];
}

/**
 * Opens the first candidate whose directory can be created and whose file can
 * be written. Returns `null` when none can be, leaving the console as the only
 * sink rather than failing the launch.
 */
export function openLogFile(): LogFile | null {
    for (const path of candidates()) {
        try {
            mkdirSync(dirname(path), { recursive: true });
            const sink = getRotatingFileSink(path, { maxSize: MAX_SIZE, maxFiles: MAX_FILES });
            return { sink, path };
        } catch {
            // Not writable (no permission, read-only mount): try the next one.
        }
    }
    return null;
}
