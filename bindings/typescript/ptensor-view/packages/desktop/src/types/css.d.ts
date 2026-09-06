// Stylesheets are imported with `with { type: 'text' }` and injected as a
// <style> tag, so to TypeScript they are plain strings.
declare module '*.css' {
    const content: string;
    export default content;
}
