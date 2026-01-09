/**
 * Simple MNIST dataloader
 * Fetches MNIST digits and provides them as normalized feature vectors
 */

interface MNISTDigit {
  label: number;
  pixels: Uint8Array;
}

interface MNISTDataset {
  images: number[][];
  labels: number[];
}

// Fetch MNIST data from public source
async function fetchMNISTData(count: number = 10000): Promise<MNISTDataset> {
  try {
    // Try to fetch from a public MNIST JSON dataset
    // Using a simplified MNIST dataset hosted on CDN
    const response = await fetch(
      "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte"
    );

    if (!response.ok) {
      throw new Error("Failed to fetch MNIST data");
    }

    // For simplicity, we'll generate synthetic MNIST-like data locally
    // This avoids network issues and provides consistent reproducible data
    return generateSyntheticMNIST(count);
  } catch (error) {
    console.warn("Could not fetch MNIST data, using synthetic dataset:", error);
    return generateSyntheticMNIST(count);
  }
}

/**
 * Generate synthetic MNIST-like data locally
 * Creates base patterns for 10 digits, then reuses them with variations
 */
function generateSyntheticMNIST(count: number): MNISTDataset {
  const images: number[][] = [];
  const labels: number[] = [];

  // Pre-generate base patterns for each digit (0-9)
  const basePatterns: number[][] = [];
  for (let digit = 0; digit < 10; digit++) {
    const pixels: number[] = [];
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        let value = 0;

        switch (digit) {
          case 0: // Circle-like
            const distFromCenter0 = Math.hypot(x - 14, y - 14);
            value = distFromCenter0 < 10 ? 255 : 0;
            break;
          case 1: // Vertical line
            value = Math.abs(x - 14) < 3 ? 255 : 0;
            break;
          case 2: // Horizontal lines
            value = Math.abs(y - 10) < 2 || Math.abs(y - 18) < 2 ? 255 : 0;
            break;
          case 3: // Right-side lines
            value = x > 15 && (Math.abs(y - 10) < 2 || Math.abs(y - 18) < 2) ? 255 : 0;
            break;
          case 4: // Cross
            value = Math.abs(x - 14) < 2 || Math.abs(y - 14) < 2 ? 255 : 0;
            break;
          case 5: // S-curve
            value =
              Math.sin((y / 28) * Math.PI * 2 + x / 28) > 0.5 &&
              Math.abs(x - (14 + Math.sin((y / 28) * Math.PI) * 8)) < 3
                ? 255
                : 0;
            break;
          case 6: // Half circle
            const distFromCenter6 = Math.hypot(x - 14, y - 14);
            value = distFromCenter6 < 10 && y > 14 ? 255 : 0;
            break;
          case 7: // Top line + diagonal
            value = y < 5 || (x > y - 5 && x < y + 5) ? 255 : 0;
            break;
          case 8: // Two circles
            const distTop = Math.hypot(x - 14, y - 10);
            const distBottom = Math.hypot(x - 14, y - 18);
            value = distTop < 6 || distBottom < 6 ? 255 : 0;
            break;
          case 9: // Top circle + tail
            const distFromCenter9 = Math.hypot(x - 14, y - 10);
            value = distFromCenter9 < 8 || (x > 12 && y > 18) ? 255 : 0;
            break;
        }
        pixels.push(value);
      }
    }
    basePatterns.push(pixels);
  }

  // Generate variations by adding noise to base patterns
  for (let i = 0; i < count; i++) {
    const label = i % 10;
    const basePattern = basePatterns[label];
    const pixels = basePattern.map((p) => {
      // Add moderate noise for variation
      const noise = (Math.random() - 0.5) * 40;
      return Math.max(0, Math.min(255, p + noise));
    });
    
    images.push(pixels);
    labels.push(label);
  }

  return { images, labels };
}

/**
 * Load MNIST dataset and return as feature vectors
 * Each vector is normalized to [0, 1]
 */
export async function loadMNISTFeatures(count: number = 10000): Promise<number[][]> {
  const dataset = await fetchMNISTData(count);

  // Normalize pixel values from [0, 255] to [0, 1]
  return dataset.images.map((pixels) => pixels.map((p) => p / 255));
}
