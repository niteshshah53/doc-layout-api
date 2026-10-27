export const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/tiff",
  "image/bmp",
  "image/webp",
] as const;

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type as (typeof ACCEPTED_TYPES)[number])) {
    return "Choose a PNG, JPEG, TIFF, BMP, or WebP image.";
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return "This image is larger than the 10 MB upload limit.";
  }
  return null;
}
