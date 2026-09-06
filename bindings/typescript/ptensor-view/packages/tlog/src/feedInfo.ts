/** Where a feed listens, and whether it came up. Reported by both servers. */
export interface FeedInfo {
  host: string;
  port: number;
  listening: boolean;
  /** Why it is not listening, when it is not. */
  error?: string;
}
