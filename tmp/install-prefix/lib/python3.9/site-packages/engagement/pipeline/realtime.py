"""Reusable source-to-pipeline loop."""

from typing import Iterable, List, Optional

from engagement.visualization.overlay import draw_records


def run_packets(pipeline, packets: Iterable[object], writer=None, display: bool = False, max_frames: Optional[int] = None):
    if display:
        import cv2
    all_records: List[object] = []
    for index, packet in enumerate(packets):
        if max_frames is not None and index >= max_frames:
            break
        records = pipeline.process_frame(packet.frame, packet.timestamp_seconds, packet.source_id)
        all_records.extend(records)
        overlay = draw_records(packet.frame, records, demo=pipeline.demo)
        if writer is not None:
            writer.write(overlay)
        if display:
            cv2.imshow("Engagement reproduction", overlay)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    if display:
        cv2.destroyAllWindows()
    return all_records

