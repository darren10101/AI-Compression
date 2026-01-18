"""
Arithmetic Coding Implementation for Learned Compression.

This module provides a range-based arithmetic coder that uses the learned
discretized logistic distributions from the IDF model for optimal compression.

The coder operates on integer symbols and uses CDF values from the learned
prior distributions to achieve near-entropy compression rates.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


# Precision constants for arithmetic coding
PRECISION = 32
CODE_VALUE_BITS = 32
MAX_RANGE = (1 << CODE_VALUE_BITS) - 1
HALF = 1 << (CODE_VALUE_BITS - 1)
QUARTER = 1 << (CODE_VALUE_BITS - 2)


class DiscretizedLogisticCDF:
    """
    Compute CDF values for the discretized logistic distribution.
    
    Used to convert learned distribution parameters (mean, log_scale)
    into cumulative distribution function values for arithmetic coding.
    """
    
    @staticmethod
    def cdf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Compute CDF at point x for discretized logistic.
        
        CDF(x) = sigmoid((x + 0.5 - mean) / scale)
        
        Args:
            x: Symbol values (float, will be treated as bin centers)
            mean: Distribution means
            scale: Distribution scales (not log scale)
        
        Returns:
            CDF values in [0, 1]
        """
        # Compute CDF at upper edge of bin containing x
        z = (x + 0.5 - mean) / np.maximum(scale, 1e-7)
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def cdf_lower(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Compute CDF at lower edge of bin.
        
        CDF_lower(x) = sigmoid((x - 0.5 - mean) / scale)
        """
        z = (x - 0.5 - mean) / np.maximum(scale, 1e-7)
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def pmf(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Compute probability mass function (CDF difference).
        
        PMF(x) = CDF(x + 0.5) - CDF(x - 0.5)
        """
        cdf_upper = DiscretizedLogisticCDF.cdf(x, mean, scale)
        cdf_lower = DiscretizedLogisticCDF.cdf_lower(x, mean, scale)
        return np.maximum(cdf_upper - cdf_lower, 1e-10)


class ArithmeticEncoder:
    """
    Range-based arithmetic encoder.
    
    Encodes a sequence of symbols given their CDF values into a compressed
    bitstream. Uses 32-bit precision for range calculations.
    """
    
    def __init__(self):
        self.low = 0
        self.high = MAX_RANGE
        self.pending_bits = 0
        self.output_bits = []
    
    def encode_symbol(self, cdf_low: float, cdf_high: float):
        """
        Encode a single symbol given its CDF range.
        
        Args:
            cdf_low: CDF value at lower edge of symbol's bin
            cdf_high: CDF value at upper edge of symbol's bin
        """
        # Clamp CDF values to valid range
        cdf_low = max(0.0, min(cdf_low, 1.0 - 1e-10))
        cdf_high = max(cdf_low + 1e-10, min(cdf_high, 1.0))
        
        # Scale CDF to integer range
        range_size = self.high - self.low + 1
        
        # Calculate new interval
        new_low = self.low + int(range_size * cdf_low)
        new_high = self.low + int(range_size * cdf_high) - 1
        
        # Ensure valid range
        if new_high <= new_low:
            new_high = new_low + 1
        
        self.low = new_low
        self.high = new_high
        
        # Renormalization
        self._renormalize()
    
    def _renormalize(self):
        """Renormalize and output bits when range converges."""
        while True:
            if self.high < HALF:
                # Output 0 and pending 1s
                self._output_bit(0)
                for _ in range(self.pending_bits):
                    self._output_bit(1)
                self.pending_bits = 0
            elif self.low >= HALF:
                # Output 1 and pending 0s
                self._output_bit(1)
                for _ in range(self.pending_bits):
                    self._output_bit(0)
                self.pending_bits = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                # Middle case - increase pending
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            
            # Double the range
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
    
    def _output_bit(self, bit: int):
        """Add a bit to output."""
        self.output_bits.append(bit)
    
    def finish(self) -> bytes:
        """
        Finish encoding and return compressed bytes.
        
        Returns:
            Compressed data as bytes
        """
        # Output remaining bits to disambiguate final state
        self.pending_bits += 1
        if self.low < QUARTER:
            self._output_bit(0)
            for _ in range(self.pending_bits):
                self._output_bit(1)
        else:
            self._output_bit(1)
            for _ in range(self.pending_bits):
                self._output_bit(0)
        
        # Pad to byte boundary
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)
        
        # Convert to bytes
        output_bytes = []
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            output_bytes.append(byte)
        
        return bytes(output_bytes)


class ArithmeticDecoder:
    """
    Range-based arithmetic decoder.
    
    Decodes symbols from a compressed bitstream using CDF values
    from learned distributions.
    """
    
    def __init__(self, data: bytes):
        self.data = data
        self.bit_pos = 0
        self.low = 0
        self.high = MAX_RANGE
        
        # Initialize code value from first bits
        self.code = 0
        for _ in range(CODE_VALUE_BITS):
            self.code = (self.code << 1) | self._read_bit()
    
    def _read_bit(self) -> int:
        """Read next bit from input."""
        if self.bit_pos >= len(self.data) * 8:
            return 0  # Pad with zeros
        
        byte_idx = self.bit_pos // 8
        bit_idx = 7 - (self.bit_pos % 8)
        self.bit_pos += 1
        
        return (self.data[byte_idx] >> bit_idx) & 1
    
    def decode_symbol(self, cdf_func, mean: float, scale: float, 
                      min_val: int, max_val: int) -> int:
        """
        Decode a single symbol given CDF function and distribution parameters.
        
        Uses binary search to find the symbol whose CDF range contains
        the current code value.
        
        Args:
            cdf_func: CDF function (takes x, mean, scale)
            mean: Distribution mean
            scale: Distribution scale
            min_val: Minimum possible symbol value
            max_val: Maximum possible symbol value
        
        Returns:
            Decoded symbol value
        """
        range_size = self.high - self.low + 1
        
        # Scale code to [0, 1] range
        scaled_code = (self.code - self.low) / range_size
        
        # Binary search for symbol
        lo, hi = min_val, max_val
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cdf_mid = DiscretizedLogisticCDF.cdf_lower(
                np.array([mid], dtype=np.float64),
                np.array([mean], dtype=np.float64),
                np.array([scale], dtype=np.float64)
            )[0]
            
            if cdf_mid <= scaled_code:
                lo = mid
            else:
                hi = mid - 1
        
        symbol = lo
        
        # Update interval
        cdf_low = DiscretizedLogisticCDF.cdf_lower(
            np.array([symbol], dtype=np.float64),
            np.array([mean], dtype=np.float64),
            np.array([scale], dtype=np.float64)
        )[0]
        cdf_high = DiscretizedLogisticCDF.cdf(
            np.array([symbol], dtype=np.float64),
            np.array([mean], dtype=np.float64),
            np.array([scale], dtype=np.float64)
        )[0]
        
        # Clamp
        cdf_low = max(0.0, cdf_low)
        cdf_high = min(1.0, cdf_high)
        if cdf_high <= cdf_low:
            cdf_high = cdf_low + 1e-10
        
        new_low = self.low + int(range_size * cdf_low)
        new_high = self.low + int(range_size * cdf_high) - 1
        
        if new_high <= new_low:
            new_high = new_low + 1
        
        self.low = new_low
        self.high = new_high
        
        # Renormalization
        self._renormalize()
        
        return symbol
    
    def _renormalize(self):
        """Renormalize by reading bits when range converges."""
        while True:
            if self.high < HALF:
                pass  # MSB is 0
            elif self.low >= HALF:
                self.code -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.code -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.code = 2 * self.code + self._read_bit()


class TensorArithmeticCoder:
    """
    High-level interface for encoding/decoding tensors using arithmetic coding
    with learned discretized logistic priors.
    
    This class handles the conversion between PyTorch tensors and the
    symbol-by-symbol arithmetic coding process.
    """
    
    def __init__(self, symbol_range: int = 512):
        """
        Args:
            symbol_range: Range of possible integer values (e.g., -256 to 255 -> 512)
        """
        self.symbol_range = symbol_range
        self.min_symbol = -symbol_range // 2
        self.max_symbol = symbol_range // 2 - 1
    
    def encode_tensor(
        self,
        latent: np.ndarray,
        mean: np.ndarray,
        log_scale: np.ndarray
    ) -> bytes:
        """
        Encode a latent tensor using arithmetic coding.
        
        Args:
            latent: Integer latent values (B, C, H, W) as float32/int
            mean: Prior mean values (same shape)
            log_scale: Prior log scale values (same shape)
        
        Returns:
            Compressed bytes
        """
        encoder = ArithmeticEncoder()
        
        # Flatten tensors
        symbols = latent.flatten().astype(np.int32)
        means = mean.flatten().astype(np.float64)
        scales = np.exp(log_scale.flatten().astype(np.float64))
        
        # Clip symbols to valid range
        symbols = np.clip(symbols, self.min_symbol, self.max_symbol)
        
        for i in range(len(symbols)):
            sym = symbols[i]
            m = means[i]
            s = scales[i]
            
            # Compute CDF bounds for this symbol
            cdf_low = DiscretizedLogisticCDF.cdf_lower(
                np.array([sym], dtype=np.float64),
                np.array([m], dtype=np.float64),
                np.array([s], dtype=np.float64)
            )[0]
            cdf_high = DiscretizedLogisticCDF.cdf(
                np.array([sym], dtype=np.float64),
                np.array([m], dtype=np.float64),
                np.array([s], dtype=np.float64)
            )[0]
            
            encoder.encode_symbol(cdf_low, cdf_high)
        
        return encoder.finish()
    
    def decode_tensor(
        self,
        data: bytes,
        shape: Tuple[int, ...],
        mean: np.ndarray,
        log_scale: np.ndarray
    ) -> np.ndarray:
        """
        Decode a latent tensor from compressed bytes.
        
        Args:
            data: Compressed bytes
            shape: Output tensor shape (B, C, H, W)
            mean: Prior mean values (same shape as output)
            log_scale: Prior log scale values (same shape)
        
        Returns:
            Decoded tensor as float32 numpy array
        """
        decoder = ArithmeticDecoder(data)
        
        means = mean.flatten().astype(np.float64)
        scales = np.exp(log_scale.flatten().astype(np.float64))
        
        num_symbols = int(np.prod(shape))
        symbols = np.zeros(num_symbols, dtype=np.float32)
        
        for i in range(num_symbols):
            m = means[i]
            s = scales[i]
            
            symbols[i] = decoder.decode_symbol(
                DiscretizedLogisticCDF.cdf,
                m, s,
                self.min_symbol, self.max_symbol
            )
        
        return symbols.reshape(shape)


def encode_latents_with_priors(
    latents: List[np.ndarray],
    prior_params: List[Tuple[np.ndarray, np.ndarray]],
    symbol_range: int = 1024
) -> Tuple[List[bytes], List[Tuple[int, ...]]]:
    """
    Encode multiple latent tensors using their learned priors.
    
    Args:
        latents: List of latent tensors (numpy arrays)
        prior_params: List of (mean, log_scale) tuples for each latent
        symbol_range: Range of possible symbol values
    
    Returns:
        compressed_data: List of compressed bytes for each latent
        shapes: List of original shapes
    """
    coder = TensorArithmeticCoder(symbol_range=symbol_range)
    compressed_data = []
    shapes = []
    
    for latent, (mean, log_scale) in zip(latents, prior_params):
        shapes.append(latent.shape)
        
        # Round latent to integers for compression
        latent_int = np.round(latent).astype(np.float32)
        
        compressed = coder.encode_tensor(latent_int, mean, log_scale)
        compressed_data.append(compressed)
    
    return compressed_data, shapes


def decode_latents_with_priors(
    compressed_data: List[bytes],
    shapes: List[Tuple[int, ...]],
    prior_params: List[Tuple[np.ndarray, np.ndarray]],
    symbol_range: int = 1024
) -> List[np.ndarray]:
    """
    Decode multiple latent tensors from compressed data.
    
    Args:
        compressed_data: List of compressed bytes
        shapes: List of original shapes
        prior_params: List of (mean, log_scale) tuples
        symbol_range: Range of possible symbol values
    
    Returns:
        List of decoded latent tensors
    """
    coder = TensorArithmeticCoder(symbol_range=symbol_range)
    latents = []
    
    for data, shape, (mean, log_scale) in zip(compressed_data, shapes, prior_params):
        decoded = coder.decode_tensor(data, shape, mean, log_scale)
        latents.append(decoded)
    
    return latents


# Vectorized encoder for better performance on large tensors
class VectorizedArithmeticEncoder:
    """
    Vectorized arithmetic encoder optimized for encoding large tensors.
    
    Uses batched CDF computation for better performance while maintaining
    exact arithmetic coding.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset encoder state."""
        self.low = np.uint64(0)
        self.high = np.uint64(MAX_RANGE)
        self.pending_bits = 0
        self.output_bytes = bytearray()
        self.bit_buffer = 0
        self.bits_in_buffer = 0
    
    def _output_bit(self, bit: int):
        """Add bit to output buffer."""
        self.bit_buffer = (self.bit_buffer << 1) | bit
        self.bits_in_buffer += 1
        
        if self.bits_in_buffer == 8:
            self.output_bytes.append(self.bit_buffer)
            self.bit_buffer = 0
            self.bits_in_buffer = 0
    
    def _output_bit_plus_pending(self, bit: int):
        """Output bit and all pending opposite bits."""
        self._output_bit(bit)
        opposite = 1 - bit
        for _ in range(self.pending_bits):
            self._output_bit(opposite)
        self.pending_bits = 0
    
    def encode_symbols_batch(
        self,
        symbols: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray
    ):
        """
        Encode a batch of symbols.
        
        Args:
            symbols: Integer symbols to encode
            means: Distribution means
            scales: Distribution scales (not log)
        """
        # Precompute all CDFs
        cdf_lows = DiscretizedLogisticCDF.cdf_lower(symbols, means, scales)
        cdf_highs = DiscretizedLogisticCDF.cdf(symbols, means, scales)
        
        for i in range(len(symbols)):
            self._encode_single(cdf_lows[i], cdf_highs[i])
    
    def _encode_single(self, cdf_low: float, cdf_high: float):
        """Encode single symbol given CDF bounds."""
        # Clamp CDFs
        cdf_low = max(0.0, min(cdf_low, 1.0 - 1e-10))
        cdf_high = max(cdf_low + 1e-10, min(cdf_high, 1.0))
        
        range_size = int(self.high - self.low + 1)
        
        new_low = int(self.low) + int(range_size * cdf_low)
        new_high = int(self.low) + int(range_size * cdf_high) - 1
        
        if new_high <= new_low:
            new_high = new_low + 1
        
        self.low = np.uint64(new_low)
        self.high = np.uint64(new_high)
        
        # Renormalize
        while True:
            if self.high < HALF:
                self._output_bit_plus_pending(0)
            elif self.low >= HALF:
                self._output_bit_plus_pending(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            
            self.low = np.uint64(2 * self.low)
            self.high = np.uint64(2 * self.high + 1)
    
    def finish(self) -> bytes:
        """Finish encoding and return bytes."""
        self.pending_bits += 1
        if self.low < QUARTER:
            self._output_bit_plus_pending(0)
        else:
            self._output_bit_plus_pending(1)
        
        # Flush remaining bits
        if self.bits_in_buffer > 0:
            self.bit_buffer <<= (8 - self.bits_in_buffer)
            self.output_bytes.append(self.bit_buffer)
        
        return bytes(self.output_bytes)


class VectorizedArithmeticDecoder:
    """
    Vectorized arithmetic decoder for decoding large tensors.
    """
    
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self.low = np.uint64(0)
        self.high = np.uint64(MAX_RANGE)
        
        # Initialize code
        self.code = np.uint64(0)
        for _ in range(CODE_VALUE_BITS):
            self.code = (self.code << 1) | self._read_bit()
    
    def _read_bit(self) -> int:
        """Read next bit."""
        if self.byte_pos >= len(self.data):
            return 0
        
        bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        
        return bit
    
    def decode_symbols_batch(
        self,
        num_symbols: int,
        means: np.ndarray,
        scales: np.ndarray,
        min_val: int,
        max_val: int
    ) -> np.ndarray:
        """
        Decode multiple symbols.
        
        Args:
            num_symbols: Number of symbols to decode
            means: Distribution means
            scales: Distribution scales
            min_val: Minimum symbol value
            max_val: Maximum symbol value
        
        Returns:
            Decoded symbols as numpy array
        """
        symbols = np.zeros(num_symbols, dtype=np.int32)
        
        for i in range(num_symbols):
            symbols[i] = self._decode_single(means[i], scales[i], min_val, max_val)
        
        return symbols
    
    def _decode_single(
        self,
        mean: float,
        scale: float,
        min_val: int,
        max_val: int
    ) -> int:
        """Decode single symbol."""
        range_size = int(self.high - self.low + 1)
        scaled_code = (int(self.code) - int(self.low)) / range_size
        
        # Binary search
        lo, hi = min_val, max_val
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cdf_mid = DiscretizedLogisticCDF.cdf_lower(
                np.array([mid], dtype=np.float64),
                np.array([mean], dtype=np.float64),
                np.array([scale], dtype=np.float64)
            )[0]
            
            if cdf_mid <= scaled_code:
                lo = mid
            else:
                hi = mid - 1
        
        symbol = lo
        
        # Update interval
        cdf_low = DiscretizedLogisticCDF.cdf_lower(
            np.array([symbol], dtype=np.float64),
            np.array([mean], dtype=np.float64),
            np.array([scale], dtype=np.float64)
        )[0]
        cdf_high = DiscretizedLogisticCDF.cdf(
            np.array([symbol], dtype=np.float64),
            np.array([mean], dtype=np.float64),
            np.array([scale], dtype=np.float64)
        )[0]
        
        cdf_low = max(0.0, cdf_low)
        cdf_high = min(1.0, cdf_high)
        if cdf_high <= cdf_low:
            cdf_high = cdf_low + 1e-10
        
        new_low = int(self.low) + int(range_size * cdf_low)
        new_high = int(self.low) + int(range_size * cdf_high) - 1
        
        if new_high <= new_low:
            new_high = new_low + 1
        
        self.low = np.uint64(new_low)
        self.high = np.uint64(new_high)
        
        # Renormalize
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.code -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.code -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            
            self.low = np.uint64(2 * self.low)
            self.high = np.uint64(2 * self.high + 1)
            self.code = np.uint64(2 * int(self.code) + self._read_bit())
        
        return symbol


def fast_encode_latents(
    latents: List[np.ndarray],
    prior_params: List[Tuple[np.ndarray, np.ndarray]],
    symbol_range: int = 1024,
    retain_residuals: bool = False,
) -> Tuple[List[bytes], List[Tuple[int, ...]], Optional[List[np.ndarray]]]:
    """
    Fast encoding using vectorized arithmetic encoder.
    
    Args:
        latents: List of latent tensors
        prior_params: List of (mean, log_scale) for each latent
        symbol_range: Range of symbol values
        retain_residuals: When True, also return (latent - rounded_latent) so the
            caller can restore exact float latents for lossless decoding.
    
    Returns:
        compressed_data: List of compressed bytes
        shapes: Original shapes
        residuals: Optional list of float32 residual tensors (only when
            retain_residuals=True)
    """
    min_symbol = -symbol_range // 2
    max_symbol = symbol_range // 2 - 1
    
    compressed_data = []
    shapes = []
    residuals = [] if retain_residuals else None
    
    for latent, (mean, log_scale) in zip(latents, prior_params):
        shapes.append(latent.shape)
        
        # Round to integers for arithmetic coding; keep residual for lossless path
        rounded = np.round(latent).astype(np.float64)
        if retain_residuals:
            residuals.append((latent.astype(np.float32) - rounded.astype(np.float32)))
        
        # Flatten symbols to match flattened means/scales expected by the encoder
        symbols = np.clip(rounded, min_symbol, max_symbol).flatten()
        means = mean.flatten().astype(np.float64)
        scales = np.exp(log_scale.flatten().astype(np.float64))
        
        # Encode
        encoder = VectorizedArithmeticEncoder()
        encoder.encode_symbols_batch(symbols, means, scales)
        compressed = encoder.finish()
        compressed_data.append(compressed)
    
    if retain_residuals:
        return compressed_data, shapes, residuals
    
    return compressed_data, shapes


def fast_decode_latents(
    compressed_data: List[bytes],
    shapes: List[Tuple[int, ...]],
    prior_params: List[Tuple[np.ndarray, np.ndarray]],
    symbol_range: int = 1024
) -> List[np.ndarray]:
    """
    Fast decoding using vectorized arithmetic decoder.
    
    Args:
        compressed_data: List of compressed bytes
        shapes: Original shapes
        prior_params: List of (mean, log_scale)
        symbol_range: Symbol value range
    
    Returns:
        List of decoded latent tensors
    """
    min_symbol = -symbol_range // 2
    max_symbol = symbol_range // 2 - 1
    
    latents = []
    
    for data, shape, (mean, log_scale) in zip(compressed_data, shapes, prior_params):
        num_symbols = int(np.prod(shape))
        means = mean.flatten().astype(np.float64)
        scales = np.exp(log_scale.flatten().astype(np.float64))
        
        decoder = VectorizedArithmeticDecoder(data)
        symbols = decoder.decode_symbols_batch(
            num_symbols, means, scales, min_symbol, max_symbol
        )
        
        latents.append(symbols.reshape(shape).astype(np.float32))
    
    return latents
