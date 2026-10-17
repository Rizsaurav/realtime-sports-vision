"""Example feature: a ping endpoint under the match.

``GET /api/matches/{match_id}/ping`` answers ``{"match_id": ..., "frames": N}``
with the number of frames the match holds.
"""


def register(router, ctx):
    @router.get("/ping")
    def ping(match_id: str):
        ctx.require(match_id)
        return {"match_id": match_id, "frames": len(ctx.frames(match_id))}
