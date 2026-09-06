// Logging for both ends of the app. The bun process and the webview each write
// to their own console; the logger category (not a hand-written '[tag]' prefix)
// carries the origin. The bun process adds a file sink on top (see
// ../server/logFile), which the webview cannot do.
//
// LogTape is an implementation detail of this module: the rest of the app talks
// to `AppLogger` and passes sinks around as opaque `LogSink` values, so
// swapping the backend touches this file (and the file-sink adapter) only.

import {
    configureSync,
    disposeSync,
    getConsoleSink,
    getLogger,
    type LogLevel,
    type Logger,
    parseLogLevel,
    type Sink,
} from '@logtape/logtape';

const ROOT_CATEGORY = 'ptensor-desktop';

let configured = false;

/** Lowest level a logger emits, from most to least verbose. */
export type LogLevelName = 'debug' | 'info' | 'warning' | 'error' | 'fatal';

/** Structured values interpolated into a message's `{placeholders}`. */
export type LogProperties = Record<string, unknown>;

/**
 * A destination for log records. Opaque on purpose: only the module that builds
 * one (e.g. `../server/logFile`) knows what is inside.
 */
export type LogSink = Sink;

/**
 * What the app logs through. Messages carry `{placeholders}` filled from
 * `properties`, e.g. `log.info('Listening on {port}.', { port })`.
 */
export interface AppLogger {
    debug(message: string, properties?: LogProperties): void;
    info(message: string, properties?: LogProperties): void;
    warn(message: string, properties?: LogProperties): void;
    error(message: string, properties?: LogProperties): void;
    fatal(message: string, properties?: LogProperties): void;
    /** Logger for a subsystem of this one, e.g. `log.child('protocol')`. */
    child(...category: string[]): AppLogger;
}

export interface LoggingOptions {
    /**
     * Lowest level to emit: 'debug', 'info', 'warning', 'error' or 'fatal'.
     * Anything else falls back to 'info'.
     */
    level?: string;
    /** Sinks to write to alongside the console, keyed by an id of your choice. */
    sinks?: Record<string, LogSink>;
}

/** Installs the sinks. Only the first call has an effect. */
export function configureLogging(options: LoggingOptions = {}): void {
    if (configured) {
        return;
    }
    configured = true;
    const sinks: Record<string, Sink> = { console: getConsoleSink(), ...options.sinks };
    const sinkIds = Object.keys(sinks);
    configureSync({
        sinks,
        loggers: [
            { category: ROOT_CATEGORY, sinks: sinkIds, lowestLevel: toLevel(options.level) },
            // LogTape's own diagnostics stay quiet unless something is wrong.
            { category: ['logtape', 'meta'], sinks: ['console'], lowestLevel: 'warning' },
        ],
        reset: true,
    });
}

function toLevel(level: string | undefined): LogLevel {
    if (level === undefined) {
        return 'info';
    }
    try {
        return parseLogLevel(level);
    } catch {
        return 'info';
    }
}

/** Logger for one subsystem, e.g. `getAppLogger('tensor-feed')`. */
export function getAppLogger(...category: string[]): AppLogger {
    configureLogging();
    return wrap(getLogger([ROOT_CATEGORY, ...category]));
}

/**
 * Flushes and closes the sinks. Call on shutdown, otherwise the tail of a file
 * log can be lost.
 */
export function disposeLogging(): void {
    if (!configured) {
        return;
    }
    configured = false;
    disposeSync();
}

function wrap(logger: Logger): AppLogger {
    return {
        debug: (message, properties) => logger.debug(message, properties),
        info: (message, properties) => logger.info(message, properties),
        warn: (message, properties) => logger.warn(message, properties),
        error: (message, properties) => logger.error(message, properties),
        fatal: (message, properties) => logger.fatal(message, properties),
        child: (...category) => wrap(logger.getChild(category as [string, ...string[]])),
    };
}
